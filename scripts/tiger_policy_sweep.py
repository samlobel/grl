import os
import jax
import optuna
import shutil
import argparse
import jax.numpy as jnp
import numpy as np

from jax.config import config
from jax import jit
from typing import Tuple, List
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path
from functools import partial

from grl.utils.file_system import numpyify_and_save
from grl.environment import load_spec
from grl.mdp import AbstractMDP, MDP
from grl.utils.math import reverse_softmax
from grl.utils.policy_eval import functional_solve_mdp
from grl.analytical_agent import AnalyticalAgent
from grl.environment.memory_lib import get_memory
from grl.memory import memory_cross_product
from grl.memory_iteration import mem_improvement, pi_improvement
from definitions import ROOT_DIR

def fill_in_params(req_params: List[float], pi_shape: Tuple):
    required_params_shape = pi_shape[:-1] + (pi_shape[-1] - 1, )
    params = np.zeros(pi_shape)
    params[:, :-1] = np.array(req_params).reshape(required_params_shape)
    params[:, -1] = 1 - params[:, :-1].sum(axis=-1)
    return params

def cpu_count():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()

@partial(jit, static_argnames=['gamma'])
def get_perf(pi_obs: jnp.ndarray,
                   T: jnp.ndarray,
                   R: jnp.ndarray,
                   p0: jnp.ndarray,
                   phi: jnp.ndarray, gamma: float):
    pi_state = phi @ pi_obs
    state_v, state_q = functional_solve_mdp(pi_state, T, R, gamma)
    return jnp.dot(p0, state_v)

def fixed_mi(pi: jnp.ndarray, amdp: AbstractMDP, mem_iterations: int = 30000,
             pi_iterations: int = 10000):
    rand_key = jax.random.PRNGKey(2020)
    mem_params = get_memory(0,
                            amdp.phi.shape[-1],
                            amdp.T.shape[0],
                            n_mem_states=2)
    pi_params = reverse_softmax(pi)

    agent = AnalyticalAgent(pi_params,
                            rand_key,
                            mem_params=mem_params,
                            policy_optim_alg='pi',
                            error_type='abs', value_type='q')

    agent.new_pi_over_mem()
    mem_loss = mem_improvement(agent,
                               amdp,
                               lr=1,
                               iterations=mem_iterations,
                               log_every=1000)

    amdp_mem = memory_cross_product(amdp, agent.mem_params)

    agent.reset_pi_params((amdp_mem.n_obs, amdp_mem.n_actions))

    # Now we improve our policy again
    policy_output = pi_improvement(agent,
                                   amdp_mem,
                                   lr=1,
                                   iterations=pi_iterations,
                                   log_every=1000)
    return get_perf(agent.policy, amdp_mem.T, amdp_mem.R, amdp_mem.p0, amdp_mem.phi, amdp_mem.gamma)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', default='tiger-alt-start', type=str,
                        help='name of POMDP spec; evals Pi_phi policies by default')
    parser.add_argument('--study_name', default='tiger-alt-start', type=str,
                        help='What is the study name')
    parser.add_argument('--new_study', action='store_true',
                        help='Delete existing study?')
    parser.add_argument('--n_jobs', default=1, type=int,
                        help='How many concurrent jobs do we split into?')
    parser.add_argument('--trials', default=100, type=int,
                        help='How many trials do we run?')

    args = parser.parse_args()

    config.update('jax_platform_name', 'cpu')

    spec = load_spec(args.spec,
                     memory_id=0,
                     n_mem_states=2)

    mdp = MDP(spec['T'], spec['R'], spec['p0'], spec['gamma'])
    amdp = AbstractMDP(mdp, spec['phi'])

    study_dir = Path(ROOT_DIR, 'results', 'analytical', args.study_name)
    journal_path = study_dir / "study.journal"
    logs_path = study_dir / "results.npy"

    if args.new_study:
        shutil.rmtree(study_dir)

    study_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=args.study_name,
        direction='maximize',
        storage=optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(str(journal_path))),
        # sampler=sampler,
        load_if_exists=True,
    )
    results = {}

    pi_params_shape = (amdp.phi.shape[-1], amdp.T.shape[0])
    def objective(trial: optuna.Trial):
        n_required_params = pi_params_shape[0] * (pi_params_shape[1] - 1)

        required_params = []
        for i in range(n_required_params):
            n_suggest_float_attempts = 0
            while True:
                n_suggest_float_attempts += 1
                if n_suggest_float_attempts >= 100 and n_suggest_float_attempts % 100 == 0:
                    print(f'Failed to suggest_float {n_suggest_float_attempts} in a row!?')
                try:
                    x = trial.suggest_float(str(i), low=0.0, high=1.0)
                except RuntimeError:
                    continue
                else:
                    break
            required_params.append(x)

        pi = fill_in_params(required_params, pi_params_shape)
        return fixed_mi(pi, amdp)

    n_jobs = args.n_jobs if args.n_jobs > 0 else cpu_count()
    if n_jobs > 1:
        n_trials_per_worker = list(map(len, np.array_split(np.arange(args.trials), n_jobs)))
        print(f'Starting pool with {n_jobs} workers')
        print(f'n_trials_per_worker: {n_trials_per_worker}')
        pool = Pool(n_jobs, maxtasksperchild=1)  # Each new tasks gets a fresh worker
        pool.starmap(
            study.optimize,
            zip(repeat(objective), n_trials_per_worker),
        )
        pool.close()
        pool.join()
    else:
        study.optimize(objective, n_trials=args.trials)

    required_params = [
        study.best_trial.params[key]
        for key in sorted(study.best_trial.params.keys(), key=lambda x: int(x))
    ]

    pi_params = fill_in_params(required_params, spec['Pi_phi'][0])
    results['best_trial_pi_params'] = pi_params
    results['best_trial_discrep'] = study.best_trial.value
    results['best_trial_number'] = study.best_trial.number

    print(f"Best trial number: {results['best_trial_number']}")
    print(f"Best trial discrepancy: {results['best_trial_discrep']}")
    print(f"Best pi_params found: {pi_params}")

    numpyify_and_save(logs_path, results)
    print(f"Saved results to {logs_path}.")
