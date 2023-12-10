import sys
import os
import argparse
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt

from environment import Environment, LBEnvironment
from up import up
from naive import naive
from cuub import cuub
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="choose one of [minisat | cplex_rcw | cplex_region]")
parser.add_argument('seed', help="random seed", nargs='?', default=9858, type=int)
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
utility_functions.append((u_unif, {'k0': 60}))

naive_ks = [60, 300, 600]

delta = .1
epsilon = .09
naive_epsilons = np.linspace(.1, .25, 6)

data_save_path = "dat/up_experiment_{}_{}.p".format(args.dataset, args.seed)

try: 
    data = pickle.load(open(data_save_path, 'rb'))
    print("Loading existing data ...")
except FileNotFoundError:
    data = {}

    for u in utility_functions:
        data[u_to_str(u)] = {}
        
        print("Running up_experiment on {} dataset with u={} and seed={}".format(args.dataset, u_to_str(u), args.seed))
        
        u_fn, u_params = u

        data[u_to_str(u)]['out_cuub'] = cuub(env, lambda t: u_fn(t, **u_params), delta, epsilon_min=epsilon)
        env.reset()
        data[u_to_str(u)]['out_up'] = up(env, lambda t: u_fn(t, **u_params), delta, epsilon_min=epsilon)
        env.reset()

        print("Calculating mean utilities ...")
        u_vect = np.vectorize(lambda t: u_fn(t, **u_params), otypes=[float])
        data[u_to_str(u)]['utilities'] = [np.mean(u_vect(env._runtimes[i, :])) for i in range(env._num_configs)]

    
        data[u_to_str(u)]['out_naive'] = {}
        for naive_k in naive_ks:
            data[u_to_str(u)]['out_naive'][naive_k] = {}
            for eps in naive_epsilons:
                try:
                    data[u_to_str(u)]['out_naive'][naive_k][eps] = naive(env, lambda t: u_fn(t, **u_params), eps, delta, naive_k)
                except AssertionError:
                    print("WARNING: naive {} failed on epsilon={}, u(k)={}".format(u_to_str(u), eps, u_fn(naive_k, **u_params)))
                env.reset()

    pickle.dump(data, open(data_save_path, 'wb'))


for u in utility_functions:
    u_str = u_to_str(u)    
    plt.scatter(data[u_str]['utilities'], data[u_str]['out_up']['total_times_by_config'][-1], label=r"UP", c=colors[5])
    plt.scatter(data[u_str]['utilities'], data[u_str]['out_cuub']['total_times_by_config'][-1], label=r"CUUB", c=colors[2])
    plt.yscale('log')
    plt.ylim(1e3, 1e7)
    plt.legend(loc='upper left')
    plt.xlabel("Utility of configuration", fontsize=fs['axis'])
    plt.ylabel("Time spent on configuration (seconds)", fontsize=fs['axis'])
    plt.title("{}".format(args.dataset), fontsize=fs['title'])
    plt.savefig("img/time_per_config_{}_{}_seed={}.pdf".format(args.dataset, u_str, args.seed), bbox_inches='tight')
    plt.clf()

    if u[0] is u_ll:
        xs = [data[u_str]['out_naive'][naive_ks[1]][eps]['total_time'] for eps in data[u_to_str(u)]['out_naive'][naive_ks[1]].keys()]
        ys = data[u_to_str(u)]['out_naive'][naive_ks[1]].keys()
        plt.scatter(xs, ys, c=colors[4], s=75, label="NAIVE ($\kappa$={})".format(naive_ks[1]))

    elif u[0] is u_unif:
        xs = [data[u_str]['out_naive'][naive_ks[0]][eps]['total_time'] for eps in data[u_to_str(u)]['out_naive'][naive_ks[0]].keys()]
        ys = data[u_to_str(u)]['out_naive'][naive_ks[0]].keys()
        plt.scatter(xs, ys, c=colors[1], s=75, label="NAIVE ($\kappa$={})".format(naive_ks[0]))

        xs = [data[u_str]['out_naive'][naive_ks[2]][eps]['total_time'] for eps in data[u_to_str(u)]['out_naive'][naive_ks[2]].keys()]
        ys = data[u_to_str(u)]['out_naive'][naive_ks[2]].keys()
        plt.scatter(xs, ys, c=colors[4], s=75, label="NAIVE ($\kappa$={})".format(naive_ks[2]))

    plt.plot(data[u_str]['out_up']['total_times'], data[u_str]['out_up']['epsilon_stars'], c=colors[5], linewidth=lw['main'], label="UP")
    plt.plot(data[u_str]['out_cuub']['total_times'], data[u_str]['out_cuub']['epsilon_stars'], c=colors[2], linewidth=lw['main'], label="CUUB")
    plt.ylim(0, 1)
    plt.xlim(1, 2e5)
    plt.xscale('log')
    plt.xlabel("Total time (CPU days)", fontsize=fs['axis'])
    plt.ylabel("$\epsilon$ guaranteed", fontsize=fs['axis'])
    plt.legend()
    plt.title("{}".format(args.dataset), fontsize=fs['title'])
    plt.savefig("img/epsilon_{}_{}_{}_seed={}.pdf".format(args.dataset, u_str, epsilon, args.seed), bbox_inches='tight')
    plt.clf()




