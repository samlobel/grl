"""
SARSA agent that utilizes an RNN.
Relies on haiku's recurrent API (https://dm-haiku.readthedocs.io/en/latest/api.html#recurrent).
Also based on David Tao's implementation at https://github.com/taodav/uncertainty/blob/main/unc/agents/lstm.py .
"""
import jax
# from jax.config import config
# config.update('jax_disable_jit', True)
import jax.numpy as jnp
import numpy as np
import haiku as hk
import optax
import dill
from functools import partial
from jax import random, jit, vmap
from optax import sgd, adam
from typing import Tuple
from grl import MDP
from grl.utils.batching import JaxBatch
from grl.mdp import one_hot
from .dqn_agent import DQNAgent
from . import DQNArgs, mse

# error func from David's impl
def seq_sarsa_error(q: jnp.ndarray, a: jnp.ndarray, r: jnp.ndarray, g: jnp.ndarray, q1: jnp.ndarray, next_a: jnp.ndarray):
    """
    sequential version of sarsa loss
    First axis of all tensors are the sequence length.
    :return:
    """
    q1_vals = q1[jnp.arange(next_a.shape[0]), next_a]

    target = r + g * q1_vals
    target = jax.lax.stop_gradient(target)

    q_vals = q[jnp.arange(a.shape[0]), a]
    return q_vals - target

def seq_sarsa_mc_error(q: jnp.ndarray, a: jnp.ndarray, r: jnp.ndarray, g: jnp.ndarray, next_a: jnp.ndarray):
    # here, q is MC q, but same same
    # This could be simpler if we assume gamma_terminal is on, but let's not
    shunted_discount = jnp.concatenate([jnp.ones_like(g[0:1]), g[:-1]])
    discount = jnp.cumprod(shunted_discount)

    discounted_r = r * discount
    cumulative_discounted_r = jnp.cumsum(discounted_r[::-1])[::-1]

    # If discount is 0, then cumulative_discounted_r is 0 as well, so we're safe from blowups
    # After shunting, it should never be that anyways though to be fair
    corrected_cumulative_r = cumulative_discounted_r / jnp.maximum(discount, 1e-5)
    target = jax.lax.stop_gradient(corrected_cumulative_r)

    q_vals = q[jnp.arange(a.shape[0]), a]
    return q_vals - target

def seq_sarsa_lambda_error(qtd: jnp.ndarray, qmc: jnp.ndarray, a: jnp.ndarray, stop_mc_grad: bool):
    q_vals_td = qtd[jnp.arange(a.shape[0]), a]
    q_vals_mc = qmc[jnp.arange(a.shape[0]), a]
    if stop_mc_grad:
        q_vals_mc = jax.lax.stop_gradient(q_vals_mc)

    return q_vals_td - q_vals_mc

def seq_sarsa_lambda_returns_error(q: jnp.ndarray, a: jnp.ndarray, r: jnp.ndarray,
                                   g: jnp.ndarray, q1: jnp.ndarray, next_a: jnp.ndarray,
                                   lambda_: float):
    # If scalar make into vector.
    lambda_ = jnp.ones_like(g) * lambda_
    q1_vals = q1[jnp.arange(next_a.shape[0]), next_a]

    # Work backwards to compute `G_{T-1}`, ..., `G_0`.
    def _body(acc, xs):
        returns, discounts, values, lambda_ = xs
        acc = returns + discounts * ((1 - lambda_) * values + lambda_ * acc)
        return acc, acc

    _, returns = jax.lax.scan(
        _body, q1_vals[-1], (r, g, q1_vals, lambda_), reverse=True)
        # _body, v_t[-1], (r, g, v_t, lambda_), reverse=True)

    lambda_returns = jax.lax.stop_gradient(returns)
    q_vals = q[jnp.arange(a.shape[0]), a]
    return q_vals - lambda_returns

class LSTMAgent(DQNAgent):
    def __init__(self, network: hk.Transformed, n_hidden: int, 
                 args: DQNArgs, mode: str = 'td0',
                 lambda_1: float = 1.0,  # The "target" value function lambda.
                 action_selection_head = 'td0',
                 lambda_coefficient: float = 1.0,
                 reward_scale : float =1.0,
                 stop_mc_grad_ld: bool = False,):
        # td0 mode means just training on TD0 loss. Both means train on both TD0 and TD1.
        # lambda means train on both, and then add lambda-discrepancy as aux term.
        assert mode in ('td0', 'td_lambda', 'both', 'lambda'), mode
        assert action_selection_head in ('td0', 'td_lambda'), action_selection_head

        self.stop_mc_grad_ld = stop_mc_grad_ld
        self.mode = mode
        self.action_selection_head = action_selection_head
        self.lambda_coefficient = lambda_coefficient
        self.reward_scale = reward_scale
        self.args = args
        # Constructor similar to DQNAgent except that network needs to be an RNN
        # TODO this is hardcoded 1 in David's impl - should this be an arg?
        self.batch_size = 1
        # internalize args
        self.features_shape = args.features_shape
        self.n_actions = args.n_actions
        self.gamma = args.gamma
        self.lambda_1 = lambda_1
        assert args.trunc_len is not None
        self.trunc_len = args.trunc_len # for truncated backprop through time
        self._rand_key, network_rand_key = random.split(args.rand_key)
        self.n_hidden = n_hidden
        self.init_hidden_var = args.init_hidden_var
        self.network = network
        self.reset()
        self.network_params = self.network.init(rng=network_rand_key, 
                                                x=jnp.zeros((self.batch_size, self.trunc_len, *self.features_shape), dtype=jnp.float32),
                                                h=self.hidden_state)
        if args.optimizer == "sgd":
            self.optimizer = sgd(args.alpha)
        elif args.optimizer == "adam":
            self.optimizer = adam(args.alpha)
        else:
            raise NotImplementedError(f"Unrecognized optimizer {args.optimizer}")
        self.optimizer_state = self.optimizer.init(self.network_params)
        self.eps = args.epsilon

        self.error_fn = None
        if args.algo == 'sarsa':
            self.td_error_fn = seq_sarsa_error
            self.batch_td_error_fn = vmap(self.td_error_fn)
            self.batch_mc_error_fn = vmap(seq_sarsa_mc_error)
            # TODO tech debt: unneccessary override
            if self.lambda_1 < 1.:
                self.batch_mc_error_fn = vmap(seq_sarsa_lambda_returns_error,
                                              in_axes=(0, 0, 0, 0, 0, 0, None))
            self.batch_lambda_error_fn = vmap(seq_sarsa_lambda_error,
                                                in_axes=(0, 0, 0, None))
        else:
            raise NotImplementedError(f"Unrecognized learning algorithm {args.algo}")

    def get_initial_hidden_state(self):
        """Get the initial state functionally so we can use it in update
            and still retain our current hidden state.
        """
        hs = jnp.zeros([self.batch_size, self.n_hidden])
        cs = jnp.zeros([self.batch_size, self.n_hidden])
        if self.init_hidden_var > 0.:
            self._rand_key, keys = random.split(self._rand_key, num=3)
            hs = random.normal(keys[0], shape=[self.batch_size, self.n_hidden]) * self.init_hidden_var
            cs = random.normal(keys[1], shape=[self.batch_size, self.n_hidden]) * self.init_hidden_var
        return hk.LSTMState(hidden=hs, cell=cs)

    def reset(self):
        """
        Reset LSTM hidden states.
        :return:
        """
        self.hidden_state = self.get_initial_hidden_state()

        
    def act(self, obs: jnp.ndarray, return_policy: bool = False) -> jnp.ndarray:
        """
        Get next epsilon-greedy action given a obs, using the agent's parameters.
        :param obs: (batch x time_steps x obs_size) obs to find actions for. 
        """        
        action, self.hidden_state, policy, self._rand_key = self._functional_act(obs, self.hidden_state, self._rand_key, self.network_params, self.eps)

        if return_policy:
            return action, policy
        return action
    
    @partial(jit, static_argnums=0)
    def _functional_act(self, obs, h, rand_key, network_params, epsilon):
       policy, _ , new_hidden = self._functional_policy(obs, h, network_params, epsilon)
       key, subkey = random.split(rand_key)
      
       # TODO is there a better way? for some reason the following doesn't work
       # action_choices = jnp.full((*exp_obs.shape[:-1], self.n_actions), jnp.arange(self.n_actions))
       # action = random.choice(key=subkey, a=action_choices, p=policy, axis=-1, shape=(*exp_obs.shape[:-1], 1))
       action = jnp.array([[]], dtype=jnp.int32)
       for batch in range(policy.shape[0]):
            actions_ts = []
            for ts in range(policy.shape[1]):
                key, subkey = random.split(key)
                actions_ts.append(random.choice(key=subkey, a = jnp.arange(self.n_actions), p=policy[batch][ts]))
            action = jnp.append(action, jnp.array([actions_ts]), axis=-1)
       return action, new_hidden, policy, key

    def policy(self, h: hk.LSTMState, obs: jnp.ndarray):
        """
        Get policy from agent's saved parameters at the given obs.
        :param obs: (batch x time_steps x obs_size) obs(s) to find actions for.
        """
        return self._functional_policy(obs, h, self.network_params, self.eps)

    @partial(jit, static_argnums=0)
    def _functional_policy(self, obs: jnp.ndarray, h: hk.LSTMState, network_params: hk.Params, epsilon: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        probs = jnp.zeros((*obs.shape[:-1], self.n_actions)) + epsilon / self.n_actions
        greedy_idx, qs, new_h = self._greedy_act(obs, h, network_params)
        # TODO I genuinely don't know a better/more efficient way to do this ATM - suggestions very welcome!
        for batch in range(probs.shape[0]):
            for ts in range(probs.shape[1]):
                probs = probs.at[(batch, ts, greedy_idx[batch][ts])].add(1 - epsilon)
        return probs, qs, new_h


    @partial(jit, static_argnums=0)
    def _greedy_act(self, obs: jnp.ndarray, h: hk.LSTMState, network_params: hk.Params) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get greedy actions given a obs
        :param obs: (b x time_steps x *obs.shape) obs to find actions
        :param network_params: NN parameters to use to calculate Q.
        :return: (b x time_steps) Greedy actions
        """
        if self.action_selection_head == 'td0':
            qs, new_h = self.Qs(obs, h, network_params)
        else:
            qs, new_h = self.Qs_td_lambda(obs, h, network_params)
        return jnp.argmax(qs, -1), qs, new_h


    def Qs(self, obs: jnp.ndarray, h: hk.LSTMState, network_params: hk.Params) -> jnp.ndarray:
        """
        Get all Q-values given a obs, from TD(0)
        :param obs: (b x time_steps x *obs.shape) obs to find action-values
        :param model: Optional. Potenially use another model
        :return: (b x time_steps x actions) torch.tensor full of action-values.
        """
        output = self.network.apply(network_params, obs, h)
        return output['td0'], output['cell_state']

    def Qs_td_lambda(self, obs: jnp.ndarray, h: hk.LSTMState, network_params: hk.Params) -> jnp.ndarray:
        """
        Get all Q-values given a obs, from TD(0)
        :param obs: (b x time_steps x *obs.shape) obs to find action-values
        :param model: Optional. Potenially use another model
        :return: (b x time_steps x actions) torch.tensor full of action-values.
        """
        output = self.network.apply(network_params, obs, h)
        return output['td_lambda'], output['cell_state']

    def both_Qs(self, obs: jnp.ndarray, h: hk.LSTMState, network_params: hk.Params):
        # Dict with keys [td0, td1, cell_state]
        return self.network.apply(network_params, obs, h)

    def Q(self, obs: jnp.ndarray, action: jnp.ndarray, h: hk.LSTMState, network_params: hk.Params) -> jnp.ndarray:
        """
        Get action-values given a obs and action
        :param obs: (b x time_steps x *obs.shape) obs to find action-values
        :param action: (b x time_steps) Actions for action-values
        :param network_params: NN parameters to use to calculate Q
        :return: (b x time_steps) Action-values
        """
        qs = self.Qs(obs, h, network_params)
        return qs[jnp.arange(action.shape[0]), action]

    def _loss(self, network_params: hk.Params,
             initial_hidden: hk.LSTMState,
             batch: JaxBatch,
             mode: str = 'td0'):
        #(B x T x A)
        both_qs_dict = self.both_Qs(batch.all_obs, initial_hidden, network_params)
        td0_q_all, td_lambda_q_all = both_qs_dict['td0'], both_qs_dict['td_lambda']
        # q_all, _ = self.Qs(batch.all_obs, initial_hidden, network_params)
        td0_q_s0 = td0_q_all[:, :-1, :]
        td0_q_s1 = td0_q_all[:, 1:, :]

        td_lambda_q_s0 = td_lambda_q_all[:, :-1, :]
        td_lambda_q_s1 = td_lambda_q_all[:, 1:, :]

        # td0_err
        effective_gamma = jax.lax.select(self.args.gamma_terminal, 1., self.gamma)
        effective_rewards = batch.rewards * self.reward_scale

        td0_err = self.batch_td_error_fn(td0_q_s0, batch.actions, effective_rewards,
                                      jnp.where(batch.terminals, 0., effective_gamma),
                                      td0_q_s1, batch.next_actions)
        # TODO tech debt: unify function signature
        if self.lambda_1 < 1.:
            td_lambda_err = self.batch_mc_error_fn(td_lambda_q_s0, batch.actions, effective_rewards,
                                             jnp.where(batch.terminals, 0., effective_gamma),
                                             td_lambda_q_s1, batch.next_actions, self.lambda_1)
        else:
            td_lambda_err = self.batch_mc_error_fn(td_lambda_q_s0, batch.actions, effective_rewards,
                jnp.where(batch.terminals, 0., effective_gamma),
                batch.next_actions)

        lambda_err = self.batch_lambda_error_fn(
            td0_q_s0, td_lambda_q_s0, batch.actions, stop_mc_grad=self.stop_mc_grad_ld)
        td0_err, td_lambda_err, lambda_err = mse(td0_err), mse(td_lambda_err), mse(lambda_err)
        if mode == 'td0':
            main_loss =  td0_err
        elif mode == 'td_lambda':
            main_loss = td_lambda_err
        elif mode == 'both':
            # main_loss = mse(td0_err) + mse(td_lambda_err)
            main_loss = td0_err + td_lambda_err
        else:
            main_loss =  td0_err + td_lambda_err + (self.lambda_coefficient * lambda_err)

        return main_loss, {
            'td0_loss': td0_err,
            'td_lambda_loss': td_lambda_err,
            'lambda_loss': lambda_err,
        }

    @partial(jit, static_argnums=(0, 1))
    def functional_update(self,
                          mode: str, # td, both, lambda
                          network_params: hk.Params,
                          optimizer_state: hk.State,
                          hidden_state: hk.LSTMState,
                          batch: JaxBatch
                          ) -> Tuple[float, hk.Params, hk.State]:
        (loss, aux_loss), grad = jax.value_and_grad(self._loss, has_aux=True)(network_params, hidden_state, batch, mode)
        updates, optimizer_state = self.optimizer.update(grad, optimizer_state, network_params)
        network_params = optax.apply_updates(network_params, updates)

        return loss, aux_loss, network_params, optimizer_state

    def update(self, 
               batch: JaxBatch
               ) -> float:
        """
        Update given a batch of data, additionally resetting the LSTM state.
        :param batch: JaxBatch of data to process.
        :return: loss
        """
        loss, aux_loss, self.network_params, self.optimizer_state = \
            self.functional_update(self.mode, self.network_params, self.optimizer_state, self.get_initial_hidden_state(), batch)
        return loss, aux_loss
    
def train_rnn_agent(mdp: MDP,
                    agent: DQNAgent,
                    total_eps: int,
                    zero_obs = False):
    """
    Training loop for a dqn agent.
    :param mdp: mdp to train on. Currently DQN does not support AMDPs.
    :param agent: DQNAgent to train.
    :param total_steps: Number of steps to train for.
    """

    steps = 0
    num_eps = 0
    args = agent.args
    epsilon_final = args.epsilon
    anneal_steps = args.anneal_steps
    epsilon_start = args.epsilon_start
    anneal_value = (epsilon_start - epsilon_final) / anneal_steps if anneal_steps > 0 else 0

    if epsilon_start != epsilon_final:
        agent.set_epsilon(epsilon_start)


    jit_onehot = jax.jit(one_hot, static_argnames=["n"])
    if zero_obs:
        jit_onehot = jax.jit(lambda x, y: jnp.zeros((mdp.n_obs,)))
    
    avg_rewards = []
    total_rewards = []
    avg_rewards_ep = []
    total_rewards_ep = []
    avg_lengths = []
    episode_lengths = []
    losses = []
    aux_losses = {}
    pct_success = 0.
    avg_len = 0.
    while (num_eps <= total_eps):
        # episode buffers
        all_obs, all_actions, terminals, rewards = [], [], [], []
        agent.reset()
        done = False
        
        o_0, _ = mdp.reset()
        o_0_processed = jit_onehot(o_0, mdp.n_obs)
        all_obs.append(o_0_processed)
        
        # need to wrap in batch dimension
        a_0, policy = agent.act(np.array([[o_0_processed]]), return_policy=True)
        a_0 = a_0[-1][-1]
        all_actions.append(a_0)

        # TODO no truncation length
        # For PR: is this functionality that we want to make optional?
        #for _ in range(agent.trunc_len):
        while not done:
            o_1, r_0, done, _, _ = mdp.step(a_0, gamma_terminal=args.gamma_terminal)
            terminals.append(done)
            rewards.append(r_0)
            steps = steps + 1  
            
            o_1_processed = jit_onehot(o_1, mdp.n_obs)
            all_obs.append(o_1_processed)

            # need to wrap in batch dimension
            # TODO tech debt: policy
            a_1, policy = agent.act(np.array([[o_1_processed]]), return_policy=True)
            a_1 = a_1[-1][-1]
            all_actions.append(a_1)

            # if done:
            #     #print(f"Broke early after {t} steps")
            #     break
            
            o_0_processed = o_1_processed
            a_0 = a_1
            
        # print(rewards)
        batch = JaxBatch(all_obs = [all_obs],
                         obs=[all_obs[:-1]], 
                         actions=[all_actions[:-1]],
                         next_obs=[all_obs[1:]],
                         terminals=[terminals],
                         rewards=[rewards],
                         next_actions=[all_actions[1:]])

       
        loss, aux_loss = agent.update(batch)
        episode_lengths.append(len(rewards))
        avg_rewards_ep.append(np.average(rewards))
        total_rewards_ep.append(np.sum(rewards))

        if num_eps % 1000 == 0:
            # various monitoring metrics
            # TODO should I trim these down?
            rewards_good = [x for x in avg_rewards_ep if x > 0]
            rewards_bad = [x for x in avg_rewards_ep if x < 0]
            pct_success = len(rewards_good) / len(avg_rewards_ep)
            pct_fail = len(rewards_bad) / len(avg_rewards_ep)
            pct_neutral = 1 - pct_success - pct_fail
            avg_rewards.append(np.average(avg_rewards_ep))
            avg_rewards_ep = []
            total_rewards.append(np.average(total_rewards_ep))
            total_rewards_ep = []

            avg_len = np.average(np.array(episode_lengths))
            avg_lengths.append(avg_len)
            episode_lengths = []

            losses.append(loss)
            # Adds aux losses as individual entries 
            for aux_key, aux_val in aux_loss.items():
                if not aux_losses.get(aux_key):
                    aux_losses[aux_key] = []
                aux_losses[aux_key].append(aux_val)

            if args.save_path:
                # TODO import saving from branch rnn_agent
                pass
            # print(f"Step {steps} | Episode {num_eps} | Epsilon {agent.eps} | Loss {loss} | Avg Length {avg_len} | Reward {batch.rewards} | Success/Fail/Neutral {pct_success}/{pct_fail}/{pct_neutral} | Obs {batch.obs} | Q-vals {agent.Qs(batch.obs, agent.get_initial_hidden_state(), agent.network_params)[0]}")
            print(f"Step {steps} | Episode {num_eps} | Epsilon {agent.eps} | Loss {loss} | Avg Length {avg_len} | Reward {batch.rewards} | Success/Fail/Neutral {pct_success}/{pct_fail}/{pct_neutral} | \n Q-vals (TD(0)) {agent.Qs(batch.obs, agent.get_initial_hidden_state(), agent.network_params)[0]} \n Q-vals (TD(lambda)) {agent.Qs_td_lambda(batch.obs, agent.get_initial_hidden_state(), agent.network_params)[0]}")
        
        num_eps = num_eps + 1
       
        # Anneal Epsilon
        if anneal_steps > 0 and anneal_steps > num_eps:
            agent.set_epsilon(epsilon_start - anneal_value * num_eps)

    agent.reset()
    final_policy, final_q, _ = agent.policy(agent.get_initial_hidden_state(), batch.obs)
    info = {"final_pct_success": pct_success, 
            "avg_len": avg_lengths, 
            "avg_reward": avg_rewards,
            "total_reward": total_rewards,
            "loss": losses,
            "aux_loss": aux_losses,
            "final_pi": final_policy,
            "final_q": final_q}
    return info, agent
