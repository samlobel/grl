import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from jax.nn import softmax
from jax.config import config
from pathlib import Path
from collections import namedtuple

config.update('jax_platform_name', 'cpu')
np.set_printoptions(precision=4)
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams.update({'font.size': 18})

from grl.utils import load_info
from definitions import ROOT_DIR

# %%
# results_dir = Path(ROOT_DIR, 'results', 'pomdps_mi_pi')
results_dir = Path(ROOT_DIR, 'results', 'pomdps_mi_pi_q_abs')
# results_dir = Path(ROOT_DIR, 'results', 'pomdps_mi_dm')
vi_results_dir = Path(ROOT_DIR, 'results', 'pomdps_vi')
pomdp_files_dir = Path(ROOT_DIR, 'grl', 'environment', 'pomdp_files')

split_by = ['spec', 'algo', 'n_mem_states']
Args = namedtuple('args', split_by)
# this option allows us to compare to either the optimal belief state soln
# or optimal state soln. ('belief' | 'state')
compare_to = 'belief'

# spec_plot_order = ['example_7', 'slippery_tmaze_5_two_thirds_up',
#                    'tiger', 'paint.95', 'cheese.95',
#                    'network', 'shuttle.95', '4x3.95']
spec_plot_order = [
    'example_7', 'tmaze_5_two_thirds_up', 'tiger-alt', 'paint.95', 'cheese.95', 'network',
    'shuttle.95', '4x3.95'
]

spec_to_belief_state = {'tmaze_5_two_thirds_up': 'tmaze'}

# %%

all_results = {}

for results_path in results_dir.iterdir():
    if results_path.is_dir() or results_path.suffix != '.npy':
        continue
    info = load_info(results_path)

    args = info['args']
    if 'n_mem_states' not in args:
        args['n_mem_states'] = 2
    # agent = info['agent']
    init_policy_info = info['logs']['initial_policy_stats']
    init_improvement_info = info['logs']['greedy_initial_improvement_stats']
    final_mem_info = info['logs']['greedy_final_mem_stats']

    def get_perf(info: dict):
        return (info['state_vals_v'] * info['p0']).sum()

    single_res = {
        'init_policy_perf': get_perf(init_policy_info),
        'init_improvement_perf': get_perf(init_improvement_info),
        'final_mem_perf': get_perf(final_mem_info),
        'init_policy': info['logs']['initial_policy'],
        'init_improvement_policy': info['logs']['initial_improvement_policy'],
        # 'final_mem': np.array(agent.memory),
        # 'final_policy': np.array(agent.policy)
    }

    hparams = Args(*tuple(args[s] for s in split_by))

    if hparams not in all_results:
        all_results[hparams] = {}

    for k, v in single_res.items():
        if k not in all_results[hparams]:
            all_results[hparams][k] = []
        all_results[hparams][k].append(v)
    all_results[hparams]['args'] = args

for hparams, res_dict in all_results.items():
    for k, v in res_dict.items():
        if k != 'args':
            all_results[hparams][k] = np.stack(v)

# %%
# Get vi performance
if compare_to == 'belief':
    for fname in pomdp_files_dir.iterdir():
        if 'pomdp-solver-results' in fname.stem:
            for hparams in all_results.keys():
                if (fname.stem ==
                        f"{spec_to_belief_state.get(hparams.spec, hparams.spec)}-pomdp-solver-results"
                    ):
                    belief_info = load_info(fname)
                    coeffs = belief_info['coeffs']
                    max_start_vals = coeffs[belief_info['max_start_idx']]
                    all_results[hparams]['compare_perf'] = np.array(
                        [np.dot(max_start_vals, belief_info['p0'])])
                    # print(f"loaded results for {hparams.spec} from {fname}")

elif compare_to == 'state':
    for hparams, res_dict in all_results.items():
        for vi_path in vi_results_dir.iterdir():
            if hparams.spec in vi_path.name:
                vi_info = load_info(vi_path)
                all_results[hparams]['compare_perf'] = np.array([
                    (vi_info['optimal_vs'] * vi_info['p0']).sum()
                ])
else:
    raise NotImplementedError

# %%
# all_normalized_perf_results = {}
for hparams, res in all_results.items():
    max_key = 'compare_perf'
    # if max_key not in res:
    #     max_key = 'final_mem_perf'
    max_v = res[max_key]
    min_v = res['init_policy_perf']
    for k, v in res.items():
        if '_perf' in k:
            all_results[hparams][k] = (v - min_v) / (max_v - min_v)
# %%

all_table_results = {}
all_plot_results = {'x': [], 'xlabels': []}

for i, spec in enumerate(spec_plot_order):
    hparams = sorted([k for k in all_results.keys() if k.spec == spec],
                     key=lambda hp: hp.n_mem_states)

    first_res = all_results[hparams[0]]
    all_plot_results['x'].append(i)
    all_plot_results['xlabels'].append(spec)

    # we first add initial and first improvement stats
    for k, v in first_res.items():
        if 'perf' in k and k != 'final_mem_perf':
            mean = v.mean(axis=0)
            std_err = v.std(axis=0) / np.sqrt(v.shape[0])

            stripped_str = k.replace('_perf', '')
            if stripped_str not in all_plot_results:
                all_plot_results[stripped_str] = {'mean': [], 'std_err': []}
            all_plot_results[stripped_str]['mean'].append(mean)
            all_plot_results[stripped_str]['std_err'].append(std_err)

    # now we add final memory perf, for each n_mem_states
    for hparam in hparams:
        res = all_results[hparam]
        for k, v in res.items():
            if k == 'final_mem_perf':
                mean = v.mean(axis=0)
                std_err = v.std(axis=0) / np.sqrt(v.shape[0])
                stripped_str = k.replace('_perf', '')
                mem_label = f"mem, |ms| = {hparam.n_mem_states}"
                if mem_label not in all_plot_results:
                    all_plot_results[mem_label] = {'mean': [], 'std_err': []}
                all_plot_results[mem_label]['mean'].append(mean)
                all_plot_results[mem_label]['std_err'].append(std_err)

ordered_plot = []
# ordered_plot.append(('init_policy', all_plot_results['init_policy']))
ordered_plot.append(('init_improvement', all_plot_results['init_improvement']))
for k in sorted(all_plot_results.keys()):
    if 'mem' in k:
        ordered_plot.append((k, all_plot_results[k]))
# ordered_plot.append(('state_optimal', all_plot_results['vi']))

# %%

# %%
def maybe_spec_map(id: str):
    spec_map = {
        '4x3.95': '4x3',
        'cheese.95': 'cheese',
        'paint.95': 'paint',
        'shuttle.95': 'shuttle',
        'example_7': 'ex. 7',
        'tmaze_5_two_thirds_up': 'tmaze',
        'tiger-alt': 'tiger'
    }
    if id not in spec_map:
        return id
    return spec_map[id]

group_width = 1
bar_width = group_width / (len(ordered_plot) + 2)
fig, ax = plt.subplots(figsize=(12, 6))

x = np.array(all_plot_results['x'])
xlabels = [maybe_spec_map(l) for l in all_plot_results['xlabels']]

for i, (label, plot_dict) in enumerate(ordered_plot):
    ax.bar(x + (i + 1) * bar_width,
           plot_dict['mean'],
           bar_width,
           yerr=plot_dict['std_err'],
           label=label)
ax.set_ylim([0, 1])
ax.set_ylabel(f'Relative Performance\n (w.r.t. optimal {compare_to} & initial policy)')
ax.set_xticks(x + group_width / 2)
ax.set_xticklabels(xlabels)
ax.legend(bbox_to_anchor=(0.7, 0.6), framealpha=0.95)
ax.set_title("Performance of Memory Iteration in POMDPs")

downloads = Path().home() / 'Downloads'
fig_path = downloads / f"{results_dir.stem}.pdf"
fig.savefig(fig_path)
