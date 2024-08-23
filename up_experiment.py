import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

from environment import Environment, LBEnvironment
from coup import oup
from up import up
from naive import naive
from utils import ensure_directory, u_ll, u_unif, u_to_str, lw, fs, colors

if __name__ == "__main__":

    ensure_directory('dat')
    ensure_directory('img')

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="choose one of [minisat | cplex_rcw | cplex_region]")
    parser.add_argument('--seed', help="random seed", nargs='?', default=987, type=int)
    parser.add_argument('--dataseed', help="data seed", nargs='?', default=520, type=int)
    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.dataset == "minisat":
        env = LBEnvironment('icar/dataset_icar/minisat_cnfuzzdd/measurements.dump', 900)
    elif args.dataset == "cplex_rcw":
        env = Environment('icar/dataset_icar/cplex_rcw/cplex_rcw_rt_seed{}.npy'.format(args.dataseed), 10000)
    elif args.dataset == "cplex_region":
        env = Environment('icar/dataset_icar/cplex_region/cplex_region_rt_seed{}.npy'.format(args.dataseed), 10000)
    else:
        print("\nERROR: dataset should be one of [minisat | cplex_rcw | cplex_region]\n")
        exit()

    utility_functions = []
    utility_functions.append((u_ll, {'k0': 60, 'a': 1}))
    utility_functions.append((u_unif, {'k0': 60}))

    # parameters from UP experiments:
    delta = .1
    epsilon = .09
    naive_epsilons = np.linspace(.25, .1, 6) # make sure decreasing 
    naive_ks = [60, 300, 600]

    data_save_path = "dat/up_experiment_{}_{}_{}.p".format(args.dataset, args.seed, args.dataseed)

    try: 
        data = pickle.load(open(data_save_path, 'rb'))
        print("Loading existing data ...")
    except FileNotFoundError:
        data = {}

        for u in utility_functions:
            u_fn, u_params = u
            u_str = u_to_str(u)
            data[u_str] = {}
            
            print("Running up_experiment on {} dataset with u={} and seed={}-".format(args.dataset, u_str, args.seed, args.dataseed))
            data[u_str]['oup'] = oup(env, lambda t: u_fn(t, **u_params), delta, epsilon_min=epsilon, doubling_condition="old")
            env.reset()
            data[u_str]['up'] = up(env, lambda t: u_fn(t, **u_params), delta, epsilon_min=epsilon, doubling_condition="old")
            env.reset()

            print("Calculating mean utilities ...")
            u_vect = np.vectorize(lambda t: u_fn(t, **u_params), otypes=[float])
            data[u_str]['utilities'] = [np.mean(u_vect(env._runtimes[i, :])) for i in range(env.num_configs)]

            data[u_str]['naive'] = {}
            for naive_k in naive_ks:
                data[u_str]['naive'][naive_k] = {}
                for eps in naive_epsilons:
                    try:
                        data[u_str]['naive'][naive_k][eps] = naive(env, lambda t: u_fn(t, **u_params), eps, delta, naive_k)
                    except AssertionError:
                        print("WARNING: naive {} failed on epsilon={}, u(k)={}".format(u_str, eps, u_fn(naive_k, **u_params)))
                    env.reset()

        pickle.dump(data, open(data_save_path, 'wb'))

    for u in utility_functions:
        u_str = u_to_str(u)
        plt.scatter(data[u_str]['utilities'], data[u_str]['up']['total_times'][-1], label=r"UP", c=colors[5])
        plt.scatter(data[u_str]['utilities'], data[u_str]['oup']['total_times'][-1], label=r"OUP", c=colors[2])
        plt.yscale('log')
        plt.legend(fontsize=fs['legend'])
        plt.xticks([t for t in plt.xticks()[0] if t >= 0], fontsize=fs['ticks'])
        plt.yticks(fontsize=fs['ticks'])
        plt.locator_params(axis='x', nbins=6)    
        plt.xlabel("Utility of configuration", fontsize=fs['axis'])
        plt.ylabel("Time per configuration (s)", fontsize=fs['axis'])
        plt.title("{}".format(args.dataset), fontsize=fs['title'])
        plt.savefig("img/up_expr_time_per_config_{}_{}_{}_{}.pdf".format(args.seed, args.dataseed, u_str, args.dataset), bbox_inches='tight')
        plt.clf()

        if u[0] is u_ll:
            xs = [data[u_str]['naive'][naive_ks[1]][eps]['total_time'] for eps in data[u_str]['naive'][naive_ks[1]].keys()]
            ys = data[u_str]['naive'][naive_ks[1]].keys()
            plt.scatter(xs, ys, c=colors[4], s=75, label="Naive($\\kappa$={})".format(naive_ks[1]))

        elif u[0] is u_unif:
            xs = [data[u_str]['naive'][naive_ks[0]][eps]['total_time'] for eps in data[u_str]['naive'][naive_ks[0]].keys()]
            ys = data[u_str]['naive'][naive_ks[0]].keys()
            plt.scatter(xs, ys, c=colors[1], s=75, label="Naive($\\kappa$={})".format(naive_ks[0]))

            xs = [data[u_str]['naive'][naive_ks[2]][eps]['total_time'] for eps in data[u_str]['naive'][naive_ks[2]].keys()]
            ys = data[u_str]['naive'][naive_ks[2]].keys()
            plt.scatter(xs, ys, c=colors[4], s=75, label="Naive($\\kappa$={})".format(naive_ks[2]))

        plt.plot(data[u_str]['up']['total_time'], data[u_str]['up']['epsilon_stars'], c=colors[5], linewidth=lw['main'], label="UP")
        plt.plot(data[u_str]['oup']['total_time'], data[u_str]['oup']['epsilon_stars'], c=colors[2], linewidth=lw['main'], label="OUP")
        plt.ylim(0, 1)
        plt.xlim(1, 2e5)
        plt.xscale('log')
        plt.xlabel("Total time (CPU days)", fontsize=fs['axis'])
        plt.ylabel("$\\epsilon$ guaranteed", fontsize=fs['axis'])
        plt.xticks(fontsize=fs['ticks'])
        plt.yticks(fontsize=fs['ticks'])
        plt.legend(fontsize=fs['legend'])
        plt.title("{}".format(args.dataset), fontsize=fs['title'])
        plt.savefig("img/up_expr_epsilon_per_runtime_{}_{}_{}_{}.pdf".format(args.seed, args.dataseed, u_str, args.dataset), bbox_inches='tight')
        plt.clf()

        n_epsilons_up = len(data[u_str]['up']['epsilon_stars'])
        times_up = []
        curr = 0
        for i in range(n_epsilons_up):
            if curr >= len(naive_epsilons):
                break
            if data[u_str]['up']['epsilon_stars'][i] <= naive_epsilons[curr]:
                times_up.append(data[u_str]['up']['total_time'][i])
                curr += 1

        n_epsilons_coup = len(data[u_str]['oup']['epsilon_stars'])
        times_coup = []
        curr = 0
        for i in range(n_epsilons_coup):
            if curr >= len(naive_epsilons):
                break
            if data[u_str]['oup']['epsilon_stars'][i] <= naive_epsilons[curr]:
                times_coup.append(data[u_str]['oup']['total_time'][i])
                curr += 1

        if u[0] is u_ll:
            xs = data[u_str]['naive'][naive_ks[1]].keys()
            ys = [data[u_str]['naive'][naive_ks[1]][eps]['total_time'] for eps in data[u_str]['naive'][naive_ks[1]].keys()]
            plt.plot(xs, ys, c=colors[4], linewidth=lw['fat'], label="Naive($\\kappa$={})".format(naive_ks[1]))
            plt.plot(naive_epsilons, times_up, c=colors[5], linewidth=lw['fat'], label="UP")
            plt.plot(naive_epsilons, times_coup, c=colors[2], linewidth=lw['fat'], label="OUP")
            plt.legend(fontsize=fs['legend'])
            plt.xlabel("$\\epsilon$", fontsize=fs['axis'])
            plt.ylabel("Total Runtime (CPU days)", fontsize=fs['axis'])
            plt.xticks(fontsize=fs['ticks'])
            plt.yticks(fontsize=fs['ticks'])
            plt.locator_params(axis='x', nbins=6)
            plt.locator_params(axis='y', nbins=6) 
            plt.title("{}".format(args.dataset), fontsize=fs['title'])
            plt.savefig("img/up_expr_runtime_per_epsilon_{}_{}_{}_{}.pdf".format(args.seed, args.dataseed, u_str, args.dataset), bbox_inches='tight')
            plt.clf()
        else:
            for naive_k in naive_ks[:1] + naive_ks[2:]:
                xs = data[u_str]['naive'][naive_k].keys()
                ys = [data[u_str]['naive'][naive_k][eps]['total_time'] for eps in data[u_str]['naive'][naive_k].keys()]
                plt.plot(xs, ys, c=colors[4], linewidth=lw['fat'], label="Naive($\\kappa$={})".format(naive_k))
                plt.plot(naive_epsilons, times_up, c=colors[5], linewidth=lw['fat'], label="UP")
                plt.plot(naive_epsilons, times_coup, c=colors[2], linewidth=lw['fat'], label="OUP")            
                plt.legend(fontsize=fs['legend'])
                plt.xlabel("$\\epsilon$", fontsize=fs['axis'])
                plt.ylabel("Total Runtime (CPU days)", fontsize=fs['axis'])
                plt.xticks(fontsize=fs['ticks'])
                plt.yticks(fontsize=fs['ticks'])
                plt.locator_params(axis='x', nbins=6)
                plt.locator_params(axis='y', nbins=6)
                plt.title("{}".format(args.dataset), fontsize=fs['title'])
                plt.savefig("img/up_expr_runtime_per_epsilon_{}_{}_{}_{}_k={}.pdf".format(args.seed, args.dataseed, u_str, args.dataset, naive_k), bbox_inches='tight')
                plt.clf()

