import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams.update({'font.size': 18})

from jax.nn import softmax
from jax.config import config
from pathlib import Path
from collections import namedtuple

import math


from grl.utils import load_info
from definitions import ROOT_DIR

def smooth_results(results, window=1):
    return np.convolve(results, np.ones(window), 'valid') / window

def get_all_results(results_path):
    # Get all the results
    result_dicts = []
    for fname in results_path.iterdir():
        if fname.is_dir() or fname.suffix != '.npy':
            continue
        
        info = load_info(fname)
        args = info['args']
        

        try:
            df_dict = {
                'reward_per_episode': info['logs']['total_reward'],
                'loss': info['logs']['loss'],
                'environment': args['spec'],
                'seed': args['seed'],
                'mode': args['lstm_mode'],
                'action_selection_head': args.get('lstm_action_selection_head'),
                'stop_mc_grad': args.get('lstm_stop_mc_grad_ld'),
                'args': args,
                'lambda_coefficient' : args.get('lambda_coefficient'),
            }
        except:
            # import ipdb; ipdb.set_trace()
            pass
            # print('good')
        if info['logs'].get('aux_loss'):
            df_dict.update(info['logs'].get('aux_loss'))

        result_dicts.append(df_dict)
        # if df_dict['mode'] == 'td_lambda':
        #     print(df_dict['action_selection_head'], df_dict['stop_mc_grad'])

    return result_dicts


def make_line_from_data(data, smoothen):
    run_results = [datum['reward_per_episode'] for datum in data]
    average_results = np.mean(run_results, axis=0)
    std_dev_results = np.std(run_results, axis=0)
    num_runs = len(run_results)
    std_err = std_dev_results / np.sqrt(num_runs)
    if smoothen > 0:
        average_results = smooth_results(average_results, window=smoothen)
        std_err = smooth_results(std_err, window=smoothen)
    return average_results, std_err


def td_plot_from_env(env_name, result_dicts, smoothen, skip=0):
    def _filter_td_lines(result):
        # return (result['environment'] == env_name
        #         and result['lstm_mode'] == 'td0'
        #         and result['action_selection_head'] == 'td0')
        return (result['environment'] == env_name
                and result['mode'] == 'td0'
                and (result['action_selection_head'] is None or result['action_selection_head'] == 'td0')
                and result['lambda_coefficient'] == 0.1)
    def _filter_both_lines(result):
        return (result['environment'] == env_name
                and result['mode'] == 'both'
                and (result['action_selection_head'] is None or result['action_selection_head'] == 'td0')
                and result['lambda_coefficient'] == 0.1)
    def _filter_ld_lines(result):
        return (result['environment'] == env_name
                and result['mode'] == 'lambda'
                and (result['action_selection_head'] is None or result['action_selection_head'] == 'td0')
                and (result['stop_mc_grad'] is None)
                and result['lambda_coefficient'] == 0.1) # will be None if we don't have
    # def _filter_ld_lines_stopgrad(result):
    #     return (result['environment'] == env_name
    #             and result['mode'] == 'lambda'
    #             and (result['action_selection_head'] is None or result['action_selection_head'] == 'td0')
    #             and (result['stop_mc_grad'] == True)) # will be None if we don't have

    td_lines = list(filter(_filter_td_lines, result_dicts))
    both_lines = list(filter(_filter_both_lines, result_dicts))
    ld_lines = list(filter(_filter_ld_lines, result_dicts))
    # ld_stopgrad_lines = list(filter(_filter_ld_lines_stopgrad, result_dicts))


    print(f"Found {len(td_lines)} td0 lines, {len(both_lines)} both lines, {len(ld_lines)} ld lines")

    td_avg, td_err = make_line_from_data(td_lines, smoothen)
    both_avg, both_err = make_line_from_data(both_lines, smoothen)
    ld_avg, ld_err = make_line_from_data(ld_lines, smoothen)
    # ld_stopgrad_avg, ld_stopgrad_err = make_line_from_data(ld_stopgrad_lines, smoothen)
    if skip:
        td_avg, td_err = td_avg[skip:], td_err[skip:]
        both_avg, both_err = both_avg[skip:], both_err[skip:]
        ld_avg, ld_err = ld_avg[skip:], ld_err[skip:]
        # ld_stopgrad_avg, ld_stopgrad_err = ld_stopgrad_avg[skip:], ld_stopgrad_err[skip:]



    plt.plot(td_avg, label=f"TD(0)")
    plt.fill_between(np.arange(len(td_avg)), td_avg - td_err, td_avg + td_err, alpha=0.2)

    plt.plot(both_avg, label=f"TD(0) + TD(1)")
    plt.fill_between(np.arange(len(both_avg)), both_avg - both_err, both_avg + both_err, alpha=0.2)

    plt.plot(ld_avg, label=f"λ-Discrepancy")
    plt.fill_between(np.arange(len(ld_avg)), ld_avg - ld_err, ld_avg + ld_err, alpha=0.2)

    # plt.plot(ld_stopgrad_avg, label=f"Discrep (stop grad)")
    # plt.fill_between(np.arange(len(ld_stopgrad_avg)), ld_stopgrad_avg - ld_stopgrad_err, ld_stopgrad_avg + ld_stopgrad_err, alpha=0.2)

def mc_plot_from_env(env_name, result_dicts, smoothen, skip=0):
    def _filter_mc_lines(result):
        # return (result['environment'] == env_name
        #         and result['lstm_mode'] == 'td0'
        #         and result['action_selection_head'] == 'td0')
        return (result['environment'] == env_name
                and result['mode'] == 'td_lambda'
                # and (result['action_selection_head'] == 'td_lambda' or result['action_selection_head'] is None))
                and result['action_selection_head'] == 'td_lambda'
                and result['lambda_coefficient'] == 0.1)
    def _filter_both_lines(result):
        return (result['environment'] == env_name
                and result['mode'] == 'both'
                and result['action_selection_head'] == 'td_lambda'
                and result['lambda_coefficient'] == 0.1)
    def _filter_ld_lines(result):
        return (result['environment'] == env_name
                and result['mode'] == 'lambda'
                and result['action_selection_head'] == 'td_lambda'
                and result['stop_mc_grad'] is None
                and result['lambda_coefficient'] == 0.1) # will be None if we don't have
    # def _filter_ld_lines_stopgrad(result):
    #     return (result['environment'] == env_name
    #             and result['mode'] == 'lambda'
    #             and result['action_selection_head'] == 'td_lambda'
    #             and result['stop_mc_grad'] == True) # will be None if we don't have

    mc_lines = list(filter(_filter_mc_lines, result_dicts))
    both_lines = list(filter(_filter_both_lines, result_dicts))
    ld_lines = list(filter(_filter_ld_lines, result_dicts))
    # ld_stopgrad_lines = list(filter(_filter_ld_lines_stopgrad, result_dicts))


    print(f"Found {len(mc_lines)} mc lines, {len(both_lines)} both lines, {len(ld_lines)} ld lines")

    mc_avg, mc_err = make_line_from_data(mc_lines, smoothen)
    both_avg, both_err = make_line_from_data(both_lines, smoothen)
    ld_avg, ld_err = make_line_from_data(ld_lines, smoothen)
    # ld_stopgrad_avg, ld_stopgrad_err = make_line_from_data(ld_stopgrad_lines, smoothen)
    if skip:
        mc_avg, mc_err = mc_avg[skip:], mc_err[skip:]
        both_avg, both_err = both_avg[skip:], both_err[skip:]
        ld_avg, ld_err = ld_avg[skip:], ld_err[skip:]
        # ld_stopgrad_avg, ld_stopgrad_err = ld_stopgrad_avg[skip:], ld_stopgrad_err[skip:]



    plt.plot(mc_avg, label=f"TD(1)")
    plt.fill_between(np.arange(len(mc_avg)), mc_avg - mc_err, mc_avg + mc_err, alpha=0.2)

    plt.plot(both_avg, label=f"TD(0) + TD(1)")
    plt.fill_between(np.arange(len(both_avg)), both_avg - both_err, both_avg + both_err, alpha=0.2)

    plt.plot(ld_avg, label=f"λ-Discrepancy")
    plt.fill_between(np.arange(len(ld_avg)), ld_avg - ld_err, ld_avg + ld_err, alpha=0.2)

    # plt.plot(ld_stopgrad_avg, label=f"Discrep (stop grad)")
    # plt.fill_between(np.arange(len(ld_stopgrad_avg)), ld_stopgrad_avg - ld_stopgrad_err, ld_stopgrad_avg + ld_stopgrad_err, alpha=0.2)


def best_plot_from_env(env_name, result_dicts, smoothen, skip=0):
    def _filter_td_lines(result):
        # return (result['environment'] == env_name
        #         and result['lstm_mode'] == 'td0'
        #         and result['action_selection_head'] == 'td0')
        return (result['environment'] == env_name
                and result['mode'] == 'td0'
                and (result['action_selection_head'] is None or result['action_selection_head'] == 'td0')
                and result['lambda_coefficient'] == 0.1)
    def _filter_mc_lines(result):
        # return (result['environment'] == env_name
        #         and result['lstm_mode'] == 'td0'
        #         and result['action_selection_head'] == 'td0')
        return (result['environment'] == env_name
                and result['mode'] == 'td_lambda'
                # and (result['action_selection_head'] == 'td_lambda' or result['action_selection_head'] is None))
                and result['action_selection_head'] == 'td_lambda'
                and result['lambda_coefficient'] == 0.1)
    def _filter_both_lines(result):
        return (result['environment'] == env_name
                and result['mode'] == 'both'
                and result['action_selection_head'] == 'td_lambda'
                and result['lambda_coefficient'] == 0.1)
    def _filter_ld_lines(result):
        return (result['environment'] == env_name
                and result['mode'] == 'lambda'
                and result['action_selection_head'] is None
                and result['stop_mc_grad'] is None
                and result['lambda_coefficient'] == 1.0) # will be None if we don't have
    # def _filter_ld_lines_stopgrad(result):
    #     return (result['environment'] == env_name
    #             and result['mode'] == 'lambda'
    #             and result['action_selection_head'] == 'td_lambda'
    #             and result['stop_mc_grad'] == True) # will be None if we don't have
    td_lines = list(filter(_filter_td_lines, result_dicts))
    mc_lines = list(filter(_filter_mc_lines, result_dicts))
    both_lines = list(filter(_filter_both_lines, result_dicts))
    ld_lines = list(filter(_filter_ld_lines, result_dicts))
    # ld_stopgrad_lines = list(filter(_filter_ld_lines_stopgrad, result_dicts))


    print(f"Found {len(mc_lines)} mc lines, {len(both_lines)} both lines, {len(ld_lines)} ld lines")

    td_avg, td_err = make_line_from_data(td_lines, smoothen)
    mc_avg, mc_err = make_line_from_data(mc_lines, smoothen)
    both_avg, both_err = make_line_from_data(both_lines, smoothen)
    ld_avg, ld_err = make_line_from_data(ld_lines, smoothen)
    # ld_stopgrad_avg, ld_stopgrad_err = make_line_from_data(ld_stopgrad_lines, smoothen)
    if skip:
        td_avg, td_err = td_avg[skip:], td_err[skip:]
        mc_avg, mc_err = mc_avg[skip:], mc_err[skip:]
        both_avg, both_err = both_avg[skip:], both_err[skip:]
        ld_avg, ld_err = ld_avg[skip:], ld_err[skip:]
        # ld_stopgrad_avg, ld_stopgrad_err = ld_stopgrad_avg[skip:], ld_stopgrad_err[skip:]


    plt.plot(td_avg, label=f"TD")
    plt.fill_between(np.arange(len(td_avg)), td_avg - td_err, td_avg + td_err, alpha=0.2)

    plt.plot(mc_avg, label=f"MC")
    plt.fill_between(np.arange(len(mc_avg)), mc_avg - mc_err, mc_avg + mc_err, alpha=0.2)

    plt.plot(both_avg, label=f"Ours (TD Head)")
    plt.fill_between(np.arange(len(both_avg)), both_avg - both_err, both_avg + both_err, alpha=0.2)

    plt.plot(ld_avg, label=f"Ours (MC Head)")
    plt.fill_between(np.arange(len(ld_avg)), ld_avg - ld_err, ld_avg + ld_err, alpha=0.2)

    # plt.plot(ld_stopgrad_avg, label=f"Discrep (stop grad)")
    # plt.fill_between(np.arange(len(ld_stopgrad_avg)), ld_stopgrad_avg - ld_stopgrad_err, ld_stopgrad_avg + ld_stopgrad_err, alpha=0.2)


def get_title_from_spec_name(spec_name):
    spec_dict = {
        'cheese.95': 'Cheese',
        'tiger-alt-start': 'Tiger',
        'network': 'Network',
        'tmaze_5_two_thirds_up': 'T-Maze',
        '4x3.95': '4x3',
        'shuttle.95': 'Shuttle',
        'paint.95': 'Paint',
        'hallway': 'Hallway',
        'hallway2': 'Hallway2',
    }
    return spec_dict[spec_name]

def make_full_td_plot_3x3(smoothen=10, skip=0):
    # Aiming for 9 environments. Legend will go underneath.
    results_path = Path(ROOT_DIR) / "shared_grl_results" / "grl_data" / "collated_final" / "final_night_results" / "recollated_td_mc_all_envs"
    results_dict = get_all_results(results_path)
    env_list = [ # TODO: make better list 
        "cheese.95", "tiger-alt-start", "network",
        "tmaze_5_two_thirds_up", "4x3.95", "shuttle.95",
        "paint.95", "hallway", "hallway2"]
    plt.subplots(3, 3, figsize=(15, 15))
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    for i, env_name in enumerate(env_list):
        plt.subplot(3, 3, i+1)
        td_plot_from_env(env_name, results_dict, smoothen=smoothen, skip=skip)
        plt.title(get_title_from_spec_name(env_name))
        plt.xlabel("Episode")
        plt.ylabel("Return")
    plt.legend(
        # bbox_to_anchor=(3, 3),
        # bbox_transform=plt.gcf().transFigure,
        bbox_to_anchor=(-0.35, -0.3),
        # bbox_transform=plt.gcf().transFigure,
        ncols=3)
    
    # plt.show()
    save_path = Path(ROOT_DIR) / 'scripts' / 'plotting' / 'neurips_plots' / 'td_head_9_envs_3x3.png'
    plt.savefig(save_path, bbox_inches='tight')


def make_full_td_plot_2x4(smoothen=10, skip=0):
    # Aiming for 9 environments. Legend will go underneath.
    results_path = Path(ROOT_DIR) / "shared_grl_results" / "grl_data" / "collated_final" / "final_night_results" / "recollated_td_mc_all_envs"
    results_dict = get_all_results(results_path)
    env_list = [ # TODO: make better list 
        "cheese.95", "tiger-alt-start", "network",
        "tmaze_5_two_thirds_up", "4x3.95", "shuttle.95",
        "paint.95", "hallway",]
        # "hallway2"]
    plt.subplots(2, 4, figsize=(24, 12))
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    for i, env_name in enumerate(env_list):
        plt.subplot(2, 4, i+1)
        td_plot_from_env(env_name, results_dict, smoothen=smoothen, skip=skip)
        plt.title(get_title_from_spec_name(env_name))
        plt.xlabel("Episode")
        plt.ylabel("Return")
    plt.legend(
        # bbox_to_anchor=(3, 3),
        # bbox_transform=plt.gcf().transFigure,
        bbox_to_anchor=(-0.35, -0.3),
        # bbox_transform=plt.gcf().transFigure,
        ncols=4)
    
    # plt.show()
    save_path = Path(ROOT_DIR) / 'scripts' / 'plotting' / 'neurips_plots' / 'td_head_9_envs_2x4.png'
    plt.savefig(save_path, bbox_inches='tight')

def make_full_mc_plot_2x4(smoothen=10, skip=0):
    # Aiming for 9 environments. Legend will go underneath.
    results_path = Path(ROOT_DIR) / "shared_grl_results" / "grl_data" / "collated_final" / "final_night_results" / "recollated_td_mc_all_envs"
    results_dict = get_all_results(results_path)
    env_list = [ # TODO: make better list 
        "cheese.95", "tiger-alt-start", "network",
        "tmaze_5_two_thirds_up", "4x3.95", "shuttle.95",
        "paint.95", "hallway",]
        # "hallway2"]
    plt.subplots(2, 4, figsize=(24, 12))
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    for i, env_name in enumerate(env_list):
        plt.subplot(2, 4, i+1)
        mc_plot_from_env(env_name, results_dict, smoothen=smoothen, skip=skip)
        plt.title(get_title_from_spec_name(env_name))
        plt.xlabel("Episode")
        plt.ylabel("Return")
    plt.legend(
        # bbox_to_anchor=(3, 3),
        # bbox_transform=plt.gcf().transFigure,
        bbox_to_anchor=(-0.35, -0.3),
        # bbox_transform=plt.gcf().transFigure,
        ncols=4)
    
    # plt.show()
    save_path = Path(ROOT_DIR) / 'scripts' / 'plotting' / 'neurips_plots' / 'mc_head_9_envs_2x4.png'
    plt.savefig(save_path, bbox_inches='tight')



def make_full_td_plot_2x2(smoothen=10, skip=0):
    # Aiming for 9 environments. Legend will go underneath.
    results_path = Path(ROOT_DIR) / "shared_grl_results" / "grl_data" / "collated_final" / "final_night_results" / "recollated_td_mc_all_envs"
    results_dict = get_all_results(results_path)
    env_list = [ # TODO: make better list 
        "tmaze_5_two_thirds_up", "4x3.95",
        "paint.95", "hallway",]
        # "hallway2"]
    # plt.suptitle("Title Test")
    plt.subplots(2, 2, figsize=(12, 9))
    plt.subplots_adjust(hspace=0.25, wspace=0.2)
    # plt.subplots_adjust(hspace=0.25, wspace=0.2, top=0.88)
    for i, env_name in enumerate(env_list):
        plt.subplot(2, 2, i+1)
        td_plot_from_env(env_name, results_dict, smoothen=smoothen, skip=skip)
        plt.title(get_title_from_spec_name(env_name))
        if i == 0 or i == 2:
            plt.ylabel("Return")
        if i == 2 or i == 3:
            plt.xlabel("Episode (thousands)")
    plt.legend(
        # bbox_to_anchor=(3, 3),
        # bbox_transform=plt.gcf().transFigure,
        bbox_to_anchor=(0.8, -0.2),
        title='Loss Target',
        # bbox_transform=plt.gcf().transFigure,
        ncols=4)
    
    # plt.show()

    plt.suptitle("Performance Using TD(0) Head For Action Selection", y=0.96)

    save_path = Path(ROOT_DIR) / 'scripts' / 'plotting' / 'neurips_plots' / 'td_head_4_envs_2x2.png'
    plt.savefig(save_path, bbox_inches='tight')

def make_full_mc_plot_2x2(smoothen=10, skip=0):
    # Aiming for 9 environments. Legend will go underneath.
    results_path = Path(ROOT_DIR) / "shared_grl_results" / "grl_data" / "collated_final" / "final_night_results" / "recollated_td_mc_all_envs"
    results_dict = get_all_results(results_path)
    env_list = [ # TODO: make better list 
        "tmaze_5_two_thirds_up", "4x3.95",
        "paint.95", "hallway",]
        # "hallway2"]

    # plt.suptitle("Title Test")

    plt.subplots(2, 2, figsize=(12, 9))
    plt.subplots_adjust(hspace=0.25, wspace=0.2)
    # plt.subplots_adjust(hspace=0.25, wspace=0.2, top=3.)
    for i, env_name in enumerate(env_list):
        plt.subplot(2, 2, i+1)
        mc_plot_from_env(env_name, results_dict, smoothen=smoothen, skip=skip)
        plt.title(get_title_from_spec_name(env_name))
        if i == 0 or i == 2:
            plt.ylabel("Return")
        if i == 2 or i == 3:
            plt.xlabel("Episode (thousands)")
    plt.legend(
        # bbox_to_anchor=(3, 3),
        # bbox_transform=plt.gcf().transFigure,
        # bbox_to_anchor=(-0.35, -0.3),
        # bbox_to_anchor=(0.5, -0.2),
        bbox_to_anchor=(0.8, -0.2),
        title='Loss Target',
        # bbox_transform=plt.gcf().transFigure,
        ncols=4)
    
    # plt.show()
    plt.suptitle("Performance Using TD(1) Head For Action Selection", y=0.96)

    save_path = Path(ROOT_DIR) / 'scripts' / 'plotting' / 'neurips_plots' / 'mc_head_4_envs_2x2.png'
    plt.savefig(save_path, bbox_inches='tight')


def make_all_plot_2x4(smoothen=10, skip=0):
    # Aiming for 9 environments. Legend will go underneath.
    results_path = Path(ROOT_DIR) / "shared_grl_results" / "grl_data" / "collated_final" / "final_night_results" / "recollated_td_mc_all_envs"
    results_dict = get_all_results(results_path)
    env_list = [ # TODO: make better list 
        "hallway", "network", "paint.95","4x3.95", "tiger-alt-start", "shuttle.95", "cheese.95",  
        "tmaze_5_two_thirds_up",  
         ]
        # "hallway2"]
    plt.subplots(2, 4, figsize=(24, 11))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, env_name in enumerate(env_list):
        plt.subplot(2, 4, i+1)
        best_plot_from_env(env_name, results_dict, smoothen=smoothen, skip=skip)
        plt.title(get_title_from_spec_name(env_name))
        if i in (0, 4):
            plt.ylabel("Return")
        if i in (4, 5, 6, 7):
            plt.xlabel("Episode (thousands)")
    plt.legend(
        # bbox_to_anchor=(3, 3),
        # bbox_transform=plt.gcf().transFigure,
        bbox_to_anchor=(-0.35, -0.3),
        # bbox_transform=plt.gcf().transFigure,
        ncols=4)
    
    # plt.show()
    plt.suptitle("Performance Of LSTM with Different Training Objectives", y=0.96)

    save_path = Path(ROOT_DIR) / 'scripts' / 'plotting' / 'neurips_plots' / 'all_8_envs_2x4_2.png'
    plt.savefig(save_path, bbox_inches='tight')

if __name__ == '__main__':
    # make_full_td_plot_3x3(smoothen=10, skip=5)
    # make_full_td_plot_2x4(smoothen=10, skip=5)
    # make_full_mc_plot_2x4(smoothen=10, skip=5)
    # make_full_td_plot_2x2(smoothen=10, skip=5)
    # make_full_mc_plot_2x2(smoothen=10, skip=5)
    make_all_plot_2x4(smoothen=10, skip=5)
    