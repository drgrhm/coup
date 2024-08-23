import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

from environment import Environment, LBEnvironment
from up import up
from coup import oup
from utils import ensure_directory, u_ll, u_unif, u_to_str, lw, fs, color_schemes

if __name__ == "__main__":

    ensure_directory('dat')
    ensure_directory('img')

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="choose one of [minisat | cplex_rcw | cplex_region]")
    parser.add_argument('--epsilon', help="accuracy parameter", nargs='?', default=.1, type=float)
    parser.add_argument('--delta', help="failure parameter", nargs='?', default=.1, type=float)
    parser.add_argument('--seed', help="random seed", nargs='?', default=987, type=int)
    parser.add_argument('--dataseed', help="seed used for cplex data", nargs='?', default=520, type=int)
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

    u = (u_ll, {'k0': 60, 'a': 1})
    # u = (u_unif, {'k0': 60})
    u_fn, u_params = u
    u_str = u_to_str(u)


    print("Running dubcond_experiment on {} dataset with u={} and seed={}_{}".format(args.dataset, u_str, args.seed, args.dataseed))
    data_save_path = "dat/dubcond_experiment_{}_{}_{}_{}.p".format(args.seed, args.dataseed, u_str, args.dataset)

    try: 
        data = pickle.load(open(data_save_path, 'rb'))
        print("Loading existing data ...")
    except FileNotFoundError:
        data = {}
        data['up_old'] = up(env, lambda t: u_fn(t, **u_params), args.delta, epsilon_min=args.epsilon, doubling_condition="old")
        env.reset()
        data['up_new'] = up(env, lambda t: u_fn(t, **u_params), args.delta, epsilon_min=args.epsilon, doubling_condition="new")
        env.reset()
        data['oup_old'] = oup(env, lambda t: u_fn(t, **u_params), args.delta, epsilon_min=args.epsilon, m_max=env.num_instances, doubling_condition="old")
        env.reset()
        data['oup_new'] = oup(env, lambda t: u_fn(t, **u_params), args.delta, epsilon_min=args.epsilon, m_max=env.num_instances, doubling_condition="new")
        env.reset()

        print("Calculating mean utilities ...")
        u_vect = np.vectorize(lambda t: u_fn(t, **u_params), otypes=[float])
        data['utilities'] = [np.mean(u_vect(env._runtimes[i, :])) for i in range(env.num_configs)]
        pickle.dump(data, open(data_save_path, 'wb'))

    plt.scatter(data['utilities'], data['up_old']['total_times'][-1], label="UP (old)", c=color_schemes[5][0])
    plt.scatter(data['utilities'], data['up_new']['total_times'][-1], label="UP (new)", c=color_schemes[5][2])
    plt.scatter(data['utilities'], data['oup_old']['total_times'][-1], label="OUP (old)", c=color_schemes[2][0])
    plt.scatter(data['utilities'], data['oup_new']['total_times'][-1], label="OUP (new)", c=color_schemes[2][2])
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
    plt.savefig("img/dubcond_time_per_config_{}_{}_{}_{}.pdf".format(args.seed, args.dataseed, u_str, args.dataset), bbox_inches='tight')
    plt.clf()

    plt.plot(data['up_old']['total_time'], data['up_old']['epsilon_stars'], c=color_schemes[5][0], linewidth=lw['main'], label="UP (old)")
    plt.plot(data['up_new']['total_time'], data['up_new']['epsilon_stars'], c=color_schemes[5][2], linewidth=lw['main'], label="UP (new)", linestyle='--')
    plt.plot(data['oup_old']['total_time'], data['oup_old']['epsilon_stars'], c=color_schemes[2][0], linewidth=lw['main'], label="OUP (old)")
    plt.plot(data['oup_new']['total_time'], data['oup_new']['epsilon_stars'], c=color_schemes[2][2], linewidth=lw['main'], label="OUP (new)", linestyle='--')
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
    plt.savefig("img/dubcond_epsilon_{}_{}_{}_{}.pdf".format(args.seed, args.dataseed, u_str, args.dataset), bbox_inches='tight')
    plt.clf()




