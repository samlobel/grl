import argparse
import logging
import pathlib
import haiku as hk
import numpy as np
import jax
from jax import random
from jax.config import config
from time import time

from grl.environment import load_spec
from grl.utils.file_system import results_path, numpyify_and_save
from grl.run import add_tmaze_hyperparams
from grl.baselines import DQNArgs, SimpleNN, ManagedLSTM
from grl.baselines.dqn_agent import DQNAgent, train_dqn_agent
from grl.baselines.rnn_agent import LSTMAgent, train_rnn_agent
from grl import MDP, AbstractMDP

# Specifies min and max rewards, we just normalize by difference.
reward_range_dict = {
    'cheese.95': (10.0, 0),
    'tiger-alt-start': (10.0, -100.0),
    'network': (80.0, -40.0),
    'tmaze_5_two_thirds_up': (4.0, -0.1),
    'example_7': (1.0, 0.0),
    '4x3.95': (1.0, -1.0),
    'shuttle.95': (10.0, -3.0),
    'paint.95': (1.0, -1.0),
    'bridge-repair': (4018,  0),
    'hallway': (1.0,  0),
    'aloha.10': (9.0, 0.0),
    'hallway2': (1.0, 0.0),
    'machine': (0.994, -15.0),
    'mit': (1.0, -1.0),
    'tiger-grid': (1.0, -1.0),
    '4x4.95': (1.0, 0.0),
}


if __name__ == '__main__':
    start_time = time()

    # Args
    parser = argparse.ArgumentParser()
    # yapf:disable
    parser.add_argument('--spec', default='example_11', type=str,
        help='name of POMDP spec')


    # TODO more from grl.run? more of my own?

    parser.add_argument('--algo', default='dqn', type=str,
                        help='Baseline algorithm to evaluate')
    parser.add_argument('--hidden_size', default=12, type=int,
                        help='For RNNs: hidden size to use.')
    parser.add_argument('--gamma_terminal', action='store_true', default=False,
                        help='Terminate episodes early with probability (1-gamma)?')
    parser.add_argument('--trunc_len', default=10, type=int,
                        help='For RNNs: backprop truncation window size. Currently not used.')
    parser.add_argument('--num_updates', default=1e5, type=int,
                        help='Number of update steps to perform; normally dqn updates every mdp step, whereas rnn updates each episode')
    parser.add_argument('--alpha', default=0.001, type=float)
    parser.add_argument('--lambda_1', default=1., type=float,
                        help='For the MC head, what value of lambda do we use for lambda-returns?')
    parser.add_argument('--epsilon', default=0.1, type=float,
                        help='What (ending, if annealing) epsilon do we use?')
    parser.add_argument('--start_epsilon', default=None, type=float,
                        help='For epsilon annealing: What starting epsilon to use?')
    parser.add_argument('--epsilon_anneal_steps', default=0, type=int,
                        help='For epsilon annealing: anneal over how many steps?')
    parser.add_argument('--log', action='store_true',
        help='save output to logs/')
    parser.add_argument('--study_name', default=None, type=str,
        help='name of the experiment. Results saved to results/{experiment_name} directory if not None. Else, save to results directory directly.')
    parser.add_argument('--platform', default='cpu', type=str,
                        help='What platform do we run things on? (cpu | gpu)')
    parser.add_argument('--seed', default=None, type=int,
        help='seed for random number generators')
    parser.add_argument('--lstm_mode', default='td0', type=str,
        help='Either td0, td_lambda, both, or lambda. Defines lstm loss')
    parser.add_argument('--lstm_action_selection_head', default='td0', type=str,
        help='Either td0 or td_lambda. Defines which head used for action selection')

    parser.add_argument('--lambda_coefficient', default=1.0, type=float,
        help='How much weight to give the lambda discrepancy')
    parser.add_argument('--reward_scale', default=1.0, type=float,
        help='How much to scale rewards during training')
    parser.add_argument('--normalize_reward_range', action='store_true', default=False)
    parser = add_tmaze_hyperparams(parser)
    global args
    args = parser.parse_args()

    np.set_printoptions(precision=4, suppress=True)
    config.update('jax_platform_name', args.platform)
    config.update("jax_enable_x64", True)

    logging.basicConfig(format='%(message)s', level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    if args.log:
        pathlib.Path('logs').mkdir(exist_ok=True)
        rootLogger = logging.getLogger()
        mem_part = 'no_memory'
        if args.use_memory is not None and args.use_memory.isdigit() and int(args.use_memory) > 0:
            mem_part = f'memory_{args.use_memory}'
        name = f'logs/{args.spec}-{mem_part}-{time()}.log'
        rootLogger.addHandler(logging.FileHandler(name))

    rand_key = None
    if args.seed is not None:
        np.random.seed(args.seed)
        rand_key = jax.random.PRNGKey(args.seed)
    else:
        rand_key = jax.random.PRNGKey(np.random.randint(1, 10000))

    # Run
    # Get POMDP definition
    spec = load_spec(args.spec,
                     corridor_length=args.tmaze_corridor_length,
                     discount=args.tmaze_discount,
                     junction_up_pi=args.tmaze_junction_up_pi,
                     epsilon=args.epsilon)

    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    pomdp = AbstractMDP(mdp, spec['phi'])


    logging.info(f'spec:\n {args.spec}\n')
    logging.info(f'T:\n {spec["T"]}')
    logging.info(f'R:\n {spec["R"]}')
    logging.info(f'gamma: {spec["gamma"]}')
    logging.info(f'p0:\n {spec["p0"]}')
    logging.info(f'phi:\n {spec["phi"]}')

    results_path = results_path(args)
    agents_dir = results_path.parent / 'agents'
    agents_dir.mkdir(exist_ok=True)

    agents_path = agents_dir / f'{results_path.stem}'

    logs, agent = None, None
    if args.algo == 'dqn_sarsa':
        # DQN uses the MDP
        def _nn_func(x):
            module = SimpleNN(mdp.n_states, mdp.n_actions)
            return module(x)
        transformed = hk.without_apply_rng(hk.transform(_nn_func))


        agent_args = DQNArgs((mdp.n_obs,),
                             mdp.n_actions,
                             mdp.gamma,
                             rand_key,
                             algo = "sarsa",
                             epsilon = args.epsilon,
                             anneal_steps=args.epsilon_anneal_steps,
                             epsilon_start=args.start_epsilon,
                             alpha = args.alpha)
        agent = DQNAgent(transformed, agent_args)

        logs, agent = train_dqn_agent(mdp, agent, args.num_updates)
    elif args.algo == 'lstm_sarsa':
        # RNN uses the pomdp
        def _lstm_func(x, h):
            module = ManagedLSTM(args.hidden_size, pomdp.n_actions)
            return module(x, h)

        transformed = hk.without_apply_rng(hk.transform(_lstm_func))

        rand_key = random.PRNGKey(2023)
        rand_key, subkey = random.split(rand_key)
        agent_args = DQNArgs((pomdp.n_obs,),
                            pomdp.n_actions,
                            pomdp.gamma,
                            subkey,
                            algo = "sarsa",
                            trunc_len=args.trunc_len,
                            alpha=args.alpha,
                            epsilon=args.epsilon,
                            epsilon_start=args.start_epsilon,
                            anneal_steps=args.epsilon_anneal_steps,
                            gamma_terminal = args.gamma_terminal,
                            )
        reward_scale = args.reward_scale
        
        if args.normalize_reward_range:
            if args.spec not in reward_range_dict:
                raise ValueError(f"Spec {args.spec} does not have a defined reward range for normalization")
            reward_scale = 1 / (reward_range_dict[args.spec][0] - reward_range_dict[args.spec][1])
            print(f'normalizing {args.spec} by {reward_scale}')
        
        agent = LSTMAgent(transformed, args.hidden_size,
                          agent_args, mode=args.lstm_mode,
                          lambda_1=args.lambda_1,
                          action_selection_head=args.lstm_action_selection_head,
                          lambda_coefficient=args.lambda_coefficient,
                          reward_scale=reward_scale,
                          # reward_scale=args.reward_scale,
                          )

        logs, agent_args = train_rnn_agent(pomdp, agent, args.num_updates)

    else:
        raise NotImplementedError(f"Error: baseline algorithm {args.algo} not recognized")



    info = {'logs': logs, 'args': args.__dict__}

    #np.save(agents_path, agent)

    end_time = time()
    run_stats = {'start_time': start_time, 'end_time': end_time}
    info['run_stats'] = run_stats

    #print(f"Saving results to {results_path}")
    numpyify_and_save(results_path, info)
