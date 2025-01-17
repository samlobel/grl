import argparse
import logging
import pathlib
from os import listdir
import os.path
from time import time

import numpy as np
import jax
from jax.config import config
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from grl.environment import load_spec
from grl.environment.pomdp_file import POMDPFile
from grl.mdp import MDP, AbstractMDP
from grl.td_lambda import td_lambda
from grl.policy_eval import PolicyEval
from grl.memory import memory_cross_product, generate_1bit_mem_fns, generate_mem_fn
from grl.pe_grad import pe_grad
from grl.utils import pformat_vals, results_path, numpyify_and_save, amdp_get_occupancy
from grl.utils.lambda_discrep import calc_discrep_from_values
from grl.memory_iteration import run_memory_iteration
from grl.vi import value_iteration

def run_pe_algos(spec: dict,
                 method: str = 'a',
                 n_random_policies: int = 0,
                 use_grad: bool = False,
                 n_episodes: int = 500,
                 lr: float = 1.,
                 value_type: str = 'v',
                 error_type: str = 'l2'):
    """
    Runs MDP, POMDP TD, and POMDP MC evaluations on given spec using given method.
    See args in __main__ function for param details.
    """
    info = {}
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    amdp = AbstractMDP(mdp, spec['phi'])

    policies = spec['Pi_phi']
    if 'mem_params' in spec.keys() and spec['mem_params'] is not None:
        amdp = memory_cross_product(amdp, spec['mem_params'])
        policies = spec['Pi_phi_x']
    if n_random_policies > 0:
        policies = amdp.generate_random_policies(n_random_policies)
    pe = PolicyEval(amdp)
    discrepancies = [] # the discrepancy dict for each policy
    discrepancy_ids = [] # Pi_phi indices (policies) where there was a discrepancy

    for i, pi in enumerate(policies):
        logging.info(f'\n\n\n======== policy id: {i} ========')
        logging.info(f'\npi:\n {pi}')
        if 'mem_params' in spec.keys():
            logging.info(f'\nmem_params:\n {spec["mem_params"]}')
        pi_ground = amdp.get_ground_policy(pi)
        logging.info(f'\npi_ground:\n {pi_ground}')

        if method == 'a' or method == 'b':
            logging.info('\n--- Analytical ---')
            mdp_vals_a, mc_vals_a, td_vals_a = pe.run(pi)
            occupancy = amdp_get_occupancy(pi, amdp)
            pr_oa = (occupancy @ amdp.phi * pi.T)
            logging.info(f'\nmdp:\n {pformat_vals(mdp_vals_a)}')
            logging.info(f'mc*:\n {pformat_vals(mc_vals_a)}')
            logging.info(f'td:\n {pformat_vals(td_vals_a)}')

            discrep = calc_discrep_from_values(td_vals_a, mc_vals_a, error_type=error_type)
            discrep['q_sum'] = (discrep['q'] * pr_oa).sum()
            info['initial_discrep'] = discrep

            logging.info(f'\ntd-mc* discrepancy:\n {pformat_vals(discrep)}')

            # If using memory, for mc and td, also aggregate obs-mem values into
            # obs values according to visitation ratios
            if 'mem_params' in spec.keys():
                occupancy_x = amdp_get_occupancy(pi, amdp)
                n_mem_states = spec['mem_params'].shape[-1]
                n_og_obs = amdp.n_obs // n_mem_states # number of obs in the original (non cross product) amdp

                # These operations are within the cross producted space
                ob_counts_x = amdp.phi.T @ occupancy_x
                ob_sums_x = ob_counts_x.reshape(n_og_obs, n_mem_states).sum(1)
                w_x = ob_counts_x / ob_sums_x.repeat(n_mem_states)

                logging.info('\n--- Cross product info')
                logging.info(f'ob-mem occupancy:\n {ob_counts_x}')
                logging.info(f'ob-mem weights:\n {w_x}')

                logging.info('\n--- Aggregation from obs-mem values (above) to obs values (below)')
                n_actions = mc_vals_a['q'].shape[0]
                mc_vals_x = {}
                td_vals_x = {}

                mc_vals_x['v'] = (mc_vals_a['v'] * w_x).reshape(n_og_obs, n_mem_states).sum(1)
                mc_vals_x['q'] = (mc_vals_a['q'] * w_x).reshape(n_actions, n_og_obs,
                                                                n_mem_states).sum(2)
                td_vals_x['v'] = (td_vals_a['v'] * w_x).reshape(n_og_obs, n_mem_states).sum(1)
                td_vals_x['q'] = (td_vals_a['q'] * w_x).reshape(n_actions, n_og_obs,
                                                                n_mem_states).sum(2)
                # logging.info(f'\nmdp:\n {pformat_vals(mdp_vals)}')
                logging.info(f'mc*:\n {pformat_vals(mc_vals_x)}')
                logging.info(f'td:\n {pformat_vals(td_vals_x)}')

                discrep = calc_discrep_from_values(td_vals_x, mc_vals_x, error_type=error_type)
                occ_obs = (occupancy_x @ amdp.phi).reshape(n_og_obs, n_mem_states).sum(-1)
                pi_obs = (pi.T * w_x).reshape(n_actions, n_og_obs, n_mem_states).sum(-1).T
                pr_oa = (occ_obs * pi_obs.T)

                discrep['q_sum'] = (discrep['q'] * pr_oa).sum()

                logging.info(f'\ntd-mc* discrepancy:\n {pformat_vals(discrep)}')

            discrepancies.append(discrep)

            if value_type:
                discrepancy_ids.append(i)
                if use_grad:
                    learnt_params, grad_info = pe_grad(spec,
                                                       pi,
                                                       grad_type=use_grad,
                                                       value_type=value_type,
                                                       error_type=error_type,
                                                       lr=lr)
                    info['grad_info'] = grad_info

        if method == 's' or method == 'b':
            # TODO: collect sample-based run info into the info dict
            # Sampling
            logging.info('\n\n--- Sampling ---')
            # MDP
            v, q = td_lambda(
                mdp,
                pi_ground,
                lambda_=1,
                alpha=0.01,
                n_episodes=n_episodes,
            )
            mdp_vals_s = {
                'v': v,
                'q': q,
            }

            # TD(1)
            v, q = td_lambda(
                amdp,
                pi,
                lambda_=1,
                alpha=0.01,
                n_episodes=n_episodes,
            )
            mc_vals_s = {
                'v': v,
                'q': q,
            }

            # TD(0)
            v, q = td_lambda(
                amdp,
                pi,
                lambda_=0,
                alpha=0.01,
                n_episodes=n_episodes,
            )
            td_vals_s = {
                'v': v,
                'q': q,
            }

            logging.info(f'mdp:\n {pformat_vals(mdp_vals_s)}')
            logging.info(f'mc:\n {pformat_vals(mc_vals_s)}')
            logging.info(f'td:\n {pformat_vals(td_vals_s)}')
            discrep = calc_discrep_from_values(td_vals_s, mc_vals_s, error_type=error_type)
            logging.info(f'\ntd-mc* discrepancy:\n {pformat_vals(discrep)}')

    logging.info('\nTD-MC* Discrepancy ids:')
    logging.info(f'{discrepancy_ids}')
    logging.info(f'({len(discrepancy_ids)}/{len(policies)})')

    return discrepancies, info

def run_generated(dir, pomdp_id=None, mem_fn_id=None):
    # There needs to be at least one memory function that decreases the discrepancy
    # under each policy.
    # So we will track for each file, for each policy, whether a memory function has been found.

    if pomdp_id is None:
        # Runs algos on all pomdps defined in 'dir' using all 1 bit memory functions.

        # The objective is to determine whether there are any specs for which no 1 bit memory function
        # decreases an existing discrepancy.
        files = [f for f in listdir(dir) if os.path.isfile(os.path.join(dir, f))]
        results = []
        for f in reversed(files):
            results.append(run_on_file(f'{dir}/{f}'))
    else:
        results = run_on_file(f'{dir}/{pomdp_id}.POMDP', mem_fn_id)
    return results

def run_on_file(filepath, mem_fn_id=None):
    spec = POMDPFile(f'{filepath}').get_spec()
    filename = os.path.basename(filepath)
    mdp_name = os.path.splitext(filename)[0]
    tag = os.path.split(os.path.dirname(filepath))[-1]

    logging.info(f'\n\n==========================================================')
    logging.info(f'GENERATED FILE: {mdp_name}')
    logging.info(f'==========================================================')

    # Discrepancies without memory.
    # List with one discrepancy dict ('v' and 'q') per policy.
    discrepancies_no_mem, _ = run_pe_algos(spec)

    path = f'grl/results/1bit_mem_conjecture_traj_weighted/{tag}'
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    if mem_fn_id is None:
        results = []
        for mem_fn_id, T_mem in enumerate(
                tqdm(
                    generate_1bit_mem_fns(n_obs=spec['phi'].shape[-1],
                                          n_actions=spec['T'].shape[0]))):

            result = record_discrepancy_improvements(path, mdp_name, spec, mem_fn_id, T_mem,
                                                     discrepancies_no_mem)
            results.append(result)
    else:
        T_mem = generate_mem_fn(mem_fn_id,
                                n_mem_states=2,
                                n_obs=spec['phi'].shape[-1],
                                n_actions=spec['T'].shape[0])
        results = record_discrepancy_improvements(path, mdp_name, spec, mem_fn_id, T_mem,
                                                  discrepancies_no_mem)
    return results

def record_discrepancy_improvements(path, mdp_name, spec, mem_fn_id, T_mem, discrepancies_no_mem):
    """Create a file if the memory function improves the discrepancy"""
    spec['T_mem'] = T_mem # add memory
    spec['Pi_phi_x'] = [pi.repeat(2, axis=0)
                        for pi in spec['Pi_phi']] # expand policies to obs-mem space
    discrepancies_mem, _ = run_pe_algos(spec)

    # Check if this memory made the discrepancy decrease for each policy.
    # The mem and no_mem lists are in the same order of policies.
    n_policies = len(discrepancies_mem)
    mem_fn_improved_discrep = [False] * n_policies
    for policy_id in range(n_policies):
        disc_no_mem = discrepancies_no_mem[policy_id]
        disc_mem = discrepancies_mem[policy_id]

        def is_pareto_q_discrepancy_improvement(disc_mem, disc_no_mem) -> bool:
            something_improved = (~np.isclose(disc_mem['q'], disc_no_mem['q'])
                                  & (disc_mem['q'] < disc_no_mem['q'])).any()
            something_got_worse = (~np.isclose(disc_mem['q'], disc_no_mem['q'])
                                   & (disc_mem['q'] > disc_no_mem['q'])).any()
            if (something_improved and not something_got_worse):
                return True
            return False

        def is_traj_weighted_q_discrepancy_improvement(disc_mem, disc_no_mem) -> bool:
            improvement = (~np.isclose(disc_mem['q_sum'], disc_no_mem['q_sum'])
                           & (disc_mem['q_sum'] < disc_no_mem['q_sum']))
            if improvement:
                return True
            return False

        # if is_pareto_q_discrepancy_improvement(disc_mem, disc_no_mem):
        if is_traj_weighted_q_discrepancy_improvement(disc_mem, disc_no_mem):
            # Create file if discrepancy was reduced
            pathlib.Path(f'{path}/{mdp_name}_{policy_id}_{mem_fn_id}.txt').touch(exist_ok=True)
            mem_fn_improved_discrep[policy_id] = True
    return mem_fn_improved_discrep

def generate_pomdps(params):
    timestamp = str(time()).replace('.', '-')
    path = f'grl/environment/pomdp_files/generated/{timestamp}'
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    for i in range(params['n_pomdps']):
        n_s = np.random.randint(params['min_n_s'], params['max_n_s'] + 1)
        n_o = np.random.randint(params['min_n_o'], params['max_n_o'] + 1)
        n_a = np.random.randint(params['min_n_a'], params['max_n_a'] + 1)
        gamma = np.random.random()
        amdp = AbstractMDP.generate(n_s, n_a, n_o, gamma=gamma)

        content = f'# Generation timestamp: {timestamp}\n'
        content += f'# with seed: {args.seed}\n'
        content += f'# with params: {params}\n\n'

        content += f'discount: {amdp.gamma}\n'
        content += 'values: reward\n'
        content += f'states: {amdp.n_states}\n'
        content += f'actions: {amdp.n_actions}\n'
        content += f'observations: {amdp.n_obs}\n'
        content += f'start: {str(amdp.p0)[1:-1]}\n\n' # remove array brackets

        # T
        for a in range(amdp.n_actions):
            content += f'T: {a}\n'
            for row in amdp.T[a]:
                content += f'{str(row)[1:-1]}\n' # remove array brackets

            content += '\n'

        # O
        content += 'O: *\n' # phi currently same for all actions
        for row in amdp.phi:
            content += f'{str(row)[1:-1]}\n' # remove array brackets

        content += '\n'

        # R
        for a in range(amdp.n_actions):
            for m, row in enumerate(amdp.R[a]):
                for n, val in enumerate(row):
                    content += f'R: {a} : {m} : {n} : * {val}\n'

            content += '\n'

        # Pi_phi
        policies = amdp.generate_random_policies(params['n_policies'])
        for pi in policies:
            content += f'Pi_phi:\n'
            for row in pi:
                content += f'{str(row)[1:-1]}\n' # remove array brackets

            content += '\n'

        with open(f'{path}/{i}.POMDP', 'w') as f:
            f.write(content)

    return timestamp

def heatmap(spec, discrep_type='l2', num_ticks=5):
    """
    (Currently have to adjust discrep_type and num_ticks above directly)
    """
    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    amdp = AbstractMDP(mdp, spec['phi'])
    policy_eval = PolicyEval(amdp, verbose=False)

    # Run for both v and q
    value_types = ['v', 'q']
    for value_type in value_types:

        if discrep_type == 'l2':
            loss_fn = policy_eval.mse_loss
        elif discrep_type == 'max':
            loss_fn = policy_eval.max_loss

        discrepancies = []
        x = y = np.linspace(0, 1, num_ticks)
        for i in range(num_ticks):
            p = x[i]
            for j in range(num_ticks):
                q = y[j]
                pi = np.array([[p, 1 - p], [q, 1 - q], [0, 0]])
                discrepancies.append(loss_fn(pi, value_type))

                if (num_ticks * i + j + 1) % 10 == 0:
                    print(f'Calculating policy {num_ticks * i + j + 1}/{num_ticks * num_ticks}')

        ax = sns.heatmap(np.array(discrepancies).reshape((num_ticks, num_ticks)),
                         xticklabels=x.round(3),
                         yticklabels=y.round(3),
                         cmap='viridis')
        ax.invert_yaxis()
        ax.set(xlabel='2nd obs', ylabel='1st obs')
        ax.set_title(f'{args.spec}, {value_type}_values, {discrep_type}_loss')
        plt.show()

def add_tmaze_hyperparams(parser: argparse.ArgumentParser):
    # hyperparams for tmaze_hperparams
    parser.add_argument('--tmaze_corridor_length',
                        default=None,
                        type=int,
                        help='Length of corridor for tmaze_hyperparams')
    parser.add_argument('--tmaze_discount',
                        default=None,
                        type=float,
                        help='Discount rate for tmaze_hyperparams')
    parser.add_argument('--tmaze_junction_up_pi',
                        default=None,
                        type=float,
                        help='probability of traversing up at junction for tmaze_hyperparams')
    return parser

if __name__ == '__main__':
    start_time = time()

    # Args
    parser = argparse.ArgumentParser()
    # yapf:disable
    parser.add_argument('--spec', default='example_11', type=str,
        help='name of POMDP spec; evals Pi_phi policies by default')
    parser.add_argument('--run_generated', type=str,
        help='name of directory with generated pomdp files located in environment/pomdp_files/generated')
    parser.add_argument('--algo', type=str, default='pe',
        help='algorithm to run. '
             '"mi" - memory and policy improvement, '
             '"pe" - policy evaluation, '
             '"vi" - value iteration on ground-truth MDP')
    parser.add_argument('--mi_iterations', type=int, default=1,
                        help='if we do memory iteration, how many iterations of memory iterations do we do?')
    parser.add_argument('--mi_steps', type=int, default=50000,
                        help='if we do memory iteration, how many steps of memory improvement do we do per iteration?')
    parser.add_argument('--pi_steps', type=int, default=50000,
                        help='if we do memory iteration, how many steps of policy improvement do we do per iteration?')
    parser.add_argument('--policy_optim_alg', type=str, default='pi',
                        help='policy improvement algorithm to use. "pi" - policy iteration, "pg" - policy gradient, '
                             '"dm" - discrepancy maximization')
    parser.add_argument('--pomdp_id', default=None, type=int)
    parser.add_argument('--mem_fn_id', default=None, type=int)
    parser.add_argument('--method', default='a', type=str,
        help='"a"-analytical, "s"-sampling, "b"-both')
    parser.add_argument('--n_random_policies', default=0, type=int,
        help='number of random policies to eval; if set (>0), overrides Pi_phi')
    parser.add_argument('--use_memory', default=None, type=int,
        help='use memory function during policy eval if set')
    parser.add_argument('--n_mem_states', default=2, type=int,
                        help='for memory_id = 0, how many memory states do we have?')
    parser.add_argument('--use_grad', default=None, type=str,
        help='find policy ("p") or memory ("m") that minimizes any discrepancies by following gradient (currently using analytical discrepancy)')
    parser.add_argument('--value_type', default='v', type=str,
                        help='Do we use (v | q) for our discrepancies?')
    parser.add_argument('--error_type', default='l2', type=str,
                        help='Do we use (l2 | abs) for our discrepancies?')
    parser.add_argument('--lr', default=1, type=float)
    parser.add_argument('--heatmap', action='store_true',
        help='generate a policy-discrepancy heatmap for the given POMDP')
    parser.add_argument('--n_episodes', default=500, type=int,
        help='number of rollouts to run')
    parser.add_argument('--generate_pomdps', default=None, nargs=8, type=int,
        help='args: n_pomdps, n_policies, min_n_s, max_n_s, min_n_a, max_n_a, min_n_o, max_n_o; generate pomdp specs and save to environment/pomdp_files/generated/')
    parser.add_argument('--log', action='store_true',
        help='save output to logs/')
    parser.add_argument('--experiment_name', default=None, type=str,
        help='name of the experiment. Results saved to results/{experiment_name} directory if not None. Else, save to results directory directly.')

    parser.add_argument('--platform', default='cpu', type=str,
                        help='What platform do we run things on? (cpu | gpu)')
    parser.add_argument('--seed', default=None, type=int,
        help='seed for random number generators')
    parser.add_argument('-f', '--fool-ipython') # hack to allow running in ipython notebooks
    parser = add_tmaze_hyperparams(parser)
    # yapf:enable

    global args
    args = parser.parse_args()
    del args.fool_ipython

    # configs
    np.set_printoptions(precision=4, suppress=True)
    config.update('jax_platform_name', args.platform)
    config.update("jax_enable_x64", True)

    logging.basicConfig(format='%(message)s', level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    if args.log:
        pathlib.Path('logs').mkdir(exist_ok=True)
        rootLogger = logging.getLogger()
        mem_part = 'no_memory'
        if args.use_memory is not None and args.use_memory > 0:
            mem_part = f'memory_{args.use_memory}'
        if args.run_generated:
            name = f'logs/{args.run_generated}.log'
        else:
            name = f'logs/{args.spec}-{mem_part}-{time()}.log'
        rootLogger.addHandler(logging.FileHandler(name))

    rand_key = None
    if args.seed is not None:
        np.random.seed(args.seed)
        rand_key = jax.random.PRNGKey(args.seed)
    else:
        rand_key = jax.random.PRNGKey(np.random.randint(1, 10000))

    # Run
    if args.generate_pomdps:
        a = args.generate_pomdps
        params = {
            'n_pomdps': a[0],
            'n_policies': a[1],
            'min_n_s': a[2],
            'max_n_s': a[3],
            'min_n_a': a[4],
            'max_n_a': a[5],
            'min_n_o': a[6],
            'max_n_o': a[7]
        }
        timestamp = generate_pomdps(params)

        print(f'Saved generated pomdp files with timestamp: {timestamp}')
    elif args.run_generated:
        run_generated(f'grl/environment/pomdp_files/generated/{args.run_generated}',
                      pomdp_id=args.pomdp_id,
                      mem_fn_id=args.mem_fn_id)
    else:
        # Get POMDP definition
        spec = load_spec(args.spec,
                         memory_id=args.use_memory,
                         n_mem_states=args.n_mem_states,
                         corridor_length=args.tmaze_corridor_length,
                         discount=args.tmaze_discount,
                         junction_up_pi=args.tmaze_junction_up_pi)
        logging.info(f'spec:\n {args.spec}\n')
        logging.info(f'T:\n {spec["T"]}')
        logging.info(f'R:\n {spec["R"]}')
        logging.info(f'gamma: {spec["gamma"]}')
        logging.info(f'p0:\n {spec["p0"]}')
        logging.info(f'phi:\n {spec["phi"]}')
        if 'mem_params' in spec.keys():
            logging.info(f'mem_params:\n {spec["mem_params"]}')
        if 'Pi_phi_x' in spec.keys():
            logging.info(f'Pi_phi_x:\n {spec["Pi_phi_x"]}')
        if 'Pi_phi' in spec and spec['Pi_phi'] is not None:
            logging.info(f'Pi_phi:\n {spec["Pi_phi"]}')

        logging.info(f'n_episodes:\n {args.n_episodes}')

        results_path = results_path(args)

        if args.heatmap:
            heatmap(spec)
        else:
            if args.algo == 'pe':

                _, info = run_pe_algos(
                    spec,
                    args.method,
                    args.n_random_policies,
                    args.use_grad,
                    args.n_episodes,
                    lr=args.lr,
                    value_type=args.value_type,
                    error_type=args.error_type,
                )
                info['args'] = args.__dict__
            elif args.algo == 'mi':
                assert args.method == 'a'
                logs, agent = run_memory_iteration(spec,
                                                   pi_lr=args.lr,
                                                   mi_lr=args.lr,
                                                   rand_key=rand_key,
                                                   mi_iterations=args.mi_iterations,
                                                   policy_optim_alg=args.policy_optim_alg,
                                                   mi_steps=args.mi_steps,
                                                   pi_steps=args.pi_steps)

                info = {'logs': logs, 'args': args.__dict__}
                agents_dir = results_path.parent / 'agents'
                agents_dir.mkdir(exist_ok=True)

                agents_path = agents_dir / f'{results_path.stem}.pkl'
                np.save(agents_path, agent)

            elif args.algo == 'vi':
                optimal_vs = value_iteration(spec['T'], spec['R'], spec['gamma'])
                print("Optimal state values from value iteration:")
                print(optimal_vs)
                info = {'optimal_vs': optimal_vs, 'p0': spec['p0'].copy(), 'args': args.__dict__}

            else:
                raise NotImplementedError

            end_time = time()
            run_stats = {'start_time': start_time, 'end_time': end_time}
            info['run_stats'] = run_stats

            print(f"Saving results to {results_path}")
            numpyify_and_save(results_path, info)
