import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

from environment import Environment, LBEnvironment
from up import up
from coup import oup
from utils import *

ensure_directory('dat')
ensure_directory('img')

parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="choose one of [minisat | cplex_rcw | cplex_region]")
parser.add_argument('seed', help="random seed", nargs='?', default=987, type=int)
parser.add_argument('epsilon', help="accuracy parameter", nargs='?', default=.1, type=float)
parser.add_argument('delta', help="failure parameter", nargs='?', default=.1, type=float)
args = parser.parse_args()

np.random.seed(args.seed)

if args.dataset == "minisat":
    env = LBEnvironment('icar/dataset_icar/minisat_cnfuzzdd/measurements.dump', 900)
elif args.dataset == "cplex_rcw":
    env = Environment('icar/dataset_icar/cplex_rcw/cplex_rcw_rt_seed{}.npy'.format(520), 10000)
elif args.dataset == "cplex_region":
    env = Environment('icar/dataset_icar/cplex_region/cplex_region_rt_seed{}.npy'.format(520), 10000)
else:
    print("\nERROR: dataset should be one of [minisat | cplex_rcw | cplex_region]\n")
    exit()

utility_functions = []
utility_functions.append((u_ll, {'k0': 60, 'a': 1}))
# utility_functions.append((u_unif, {'k0': 60}))

data = {}
for u in utility_functions:
    u_str = u_to_str(u)
    print("Running dubcond_experiment on {} dataset with u={} and seed={}".format(args.dataset, u_str, args.seed))

    u_fn, u_params = u
    data[u_str] = {}

    data_save_path = "dat/dubcond_experiment_{}_{}_{}.p".format(u_str, args.dataset, args.seed)

    try: 
        data = pickle.load(open(data_save_path, 'rb'))
        print("Loading existing data ...")
    except FileNotFoundError:
        data = {}
        data['up_old'] = up(env, lambda t: u_fn(t, **u_params), args.delta, epsilon_min=args.epsilon, doubling_condition="old")
        env.reset()
        data['up_new'] = up(env, lambda t: u_fn(t, **u_params), args.delta, epsilon_min=args.epsilon, doubling_condition="new")
        env.reset()
        data['oup_old'] = oup(env, lambda t: u_fn(t, **u_params), args.delta, epsilon_min=args.epsilon, m_max=env.get_num_instances(), doubling_condition="old")
        env.reset()
        data['oup_new'] = oup(env, lambda t: u_fn(t, **u_params), args.delta, epsilon_min=args.epsilon, m_max=env.get_num_instances(), doubling_condition="new")
        env.reset()

        print("Calculating mean utilities ...")
        u_vect = np.vectorize(lambda t: u_fn(t, **u_params), otypes=[float])
        data['utilities'] = [np.mean(u_vect(env._runtimes[i, :])) for i in range(env._num_configs)]
        
        pickle.dump(data, open(data_save_path, 'wb'))

    plt.scatter(data['utilities'], data['up_old']['total_times_by_config'][-1], label="UP (old)", c=color_schemes[5][0])
    plt.scatter(data['utilities'], data['up_new']['total_times_by_config'][-1], label="UP (new)", c=color_schemes[5][2])
    plt.scatter(data['utilities'], data['oup_old']['total_times_by_config'][-1], label="OUP (old)", c=color_schemes[2][0])
    plt.scatter(data['utilities'], data['oup_new']['total_times_by_config'][-1], label="OUP (new)", c=color_schemes[2][2])
    plt.xticks(fontsize=fs['ticks'])
    plt.yticks(fontsize=fs['ticks'])
    plt.locator_params(axis='x', nbins=6)
    plt.yscale('log')
    legend_scale = {'minisat': 4.3, 'cplex_rcw': 2.7, 'cplex_region':1.5}
    plt.ylim(plt.ylim()[0], plt.ylim()[1] * legend_scale[args.dataset])
    plt.legend(fontsize=fs['legend'])
    plt.xlabel("Utility of configuration", fontsize=fs['axis'])
    plt.ylabel("Time per configuration (s)", fontsize=fs['axis'])
    plt.title("{}".format(args.dataset), fontsize=fs['title'])
    plt.savefig("img/dubcond_time_per_config_{}_{}_seed={}.pdf".format(args.dataset, u_str, args.seed), bbox_inches='tight')
    plt.clf()

    plt.plot(data['up_old']['total_times'], data['up_old']['epsilon_stars'], c=color_schemes[5][0], linewidth=lw['main'], label="UP (old)")
    plt.plot(data['up_new']['total_times'], data['up_new']['epsilon_stars'], c=color_schemes[5][2], linewidth=lw['main'], label="UP (new)", linestyle='--')
    plt.plot(data['oup_old']['total_times'], data['oup_old']['epsilon_stars'], c=color_schemes[2][0], linewidth=lw['main'], label="OUP (old)")
    plt.plot(data['oup_new']['total_times'], data['oup_new']['epsilon_stars'], c=color_schemes[2][2], linewidth=lw['main'], label="OUP (new)", linestyle='--')
    plt.ylim(0, 1)    
    plt.xscale('log')
    plt.xticks(fontsize=fs['ticks'])
    plt.yticks(fontsize=fs['ticks'])
    plt.xlabel("Total time (CPU days)", fontsize=fs['axis'])
    plt.ylabel("$\\epsilon$ guaranteed", fontsize=fs['axis'])    
    if args.dataset == 'cplex_rcw':
        plt.xlim(1, 2e4)
        plt.legend(fontsize=fs['legend'], loc='lower left')
    elif args.dataset == 'cplex_region':
        plt.xlim(1, 3e3)
        plt.legend(fontsize=fs['legend'], loc='lower left')
    else:
        plt.legend(fontsize=fs['legend'])
    plt.title("{}".format(args.dataset), fontsize=fs['title'])
    plt.savefig("img/dubcond_epsilon_{}_{}_{}_seed={}.pdf".format(args.dataset, u_str, args.epsilon, args.seed), bbox_inches='tight')
    plt.clf()




