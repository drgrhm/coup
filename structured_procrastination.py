""" Code taken from https://github.com/drgrhm/alg_config """

import argparse
import os
import numpy as np
# import simulated_environment
from configuration_tester import ConfigurationTester
import pickle
from utils import day_in_s
import time


def spc(env, k0, theta_multiplier, stop_time):
    """Implementation of Structured Procrastination with Confidence.
    """
    n = env.num_configs
    configs = {}  # configurations
    for i in range(n):
        configs[i] = ConfigurationTester(i, k0, theta_multiplier)

    # time_so_far = 0
    iter_count = 0
    # t0, t1 = 0, 0

    i_star = None

    results = {'i_stars': [], 'total_time': [], 'num_actives': []}
    configs_r = []
    configs_total_time = []

    # for stop_time in stop_times:
    while env.total_time < stop_time:

        i, _ = min([(cid, config.get_confidence_bound(iter_count)) for cid, config in configs.items()], key=lambda t: t[1])

        configs[i].execute_step(env, iter_count)

        if iter_count % 50000 == 0:
            num_actives = [(i, c.get_num_active()) for i, c in configs.items()]
            i_star, i_star_r = max(num_actives, key=lambda t: t[1])
            results['i_stars'].append(i_star)
            results['total_time'].append(env.total_time / day_in_s)
            results['num_actives'].append(num_actives)
            print('iter_count={}, total_time={:.4f}, stop_time={:.4f}, i_star={}'.format(iter_count, env.total_time / day_in_s, stop_time / day_in_s, i_star), flush=True)

        iter_count += 1

    # num_actives = [(i, c.get_num_active()) for i, c in configs.items()]
    # i_star, i_star_r = max(num_actives, key=lambda t: t[1])
    
    # print('reached stop_time={}. i_star={}, total_time={}'.format(stop_time, i_star, env.total_time / day_in_s), flush=True)

    # results['i_stars'].append(i_star)
    # results['total_times'].append(env.total_time)


    num_actives = [(i, c.get_num_active()) for i, c in configs.items()]
    i_star, _ = max(num_actives, key=lambda t:t[1])

    results['i_stars'].append(i_star)
    results['total_time'].append(env.total_time / day_in_s)


    return results


# def main():
#     parser = argparse.ArgumentParser(description='Executes Structured Procrastination with Confidence with a simulated environment.')
#     parser.add_argument('--k0', help='Kappa_0 from the paper', type=float, default=1.)
#     parser.add_argument('--theta-multiplier', help='Theta multiplier from the paper', type=float, default=2.)
#     parser.add_argument('--measurements-filename', help='Filename to load measurement results from', type=str, default='measurements.dump')
#     parser.add_argument('--measurements-timeout', help='Timeout (seconds) used for the measurements', type=float, default=900.)
#     parser.add_argument('--total_time_budget', help='Total time (seconds) allowed', type=float, default=24.*60.*60.*2700.)  # 2700 CPU days;
#     args = vars(parser.parse_args())

#     k0 = args['k0']
#     theta_multiplier = args['theta_multiplier']
#     results_file = args['measurements_filename']
#     timeout = args['measurements_timeout']
#     total_time_budget = args['total_time_budget']

#     try: os.mkdir('results')
#     except OSError: pass

#     print("creating simulated environment")
#     env = simulated_environment.Environment(results_file, timeout)
#     num_configs = env.get_num_configs()

#     print("running structured_procrastination_confidence")

#     step_size = int(day_in_seconds)  # CPU day, in second
#     # stop_times = list(range(step_size, 10 * int(day_in_seconds), step_size)) + list(range(10 * int(day_in_seconds), int(total_time_budget) + 1, 10 * step_size))  # check results at 1,2,3,..,9,10,20,30,... CPU days
#     stop_times = list(range(step_size, 10 * int(day_in_seconds) + 1, step_size)) + list(range(50 * int(day_in_seconds), int(total_time_budget) + 1, 50 * step_size))  # check results at 1,2,3,..,9,10,50,100,150,... CPU days

#     t0 = time.time()
#     best_config_index, configs = structured_procrastination_confidence(env, num_configs, k0, theta_multiplier, total_time_budget, stop_times)
#     t1 = time.time()

#     print("")
#     print("for total_time_budget={}".format(total_time_budget))
#     print('best_config_index={}'.format(best_config_index))

#     print("")
#     print('total runtime: ' + format_runtime(env.get_total_runtime()))
#     print('total resumed runtime: ' + format_runtime(env.get_total_resumed_runtime()))

#     print("")
#     print("Total real time to run: {}".format(t1 - t0))



# if __name__ == '__main__':
#     # np.set_printoptions(suppress=True,linewidth=np.nan,threshold=np.nan)

#     main()
