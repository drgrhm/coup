import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

from environment import Environment, LBEnvironment
from coup import coup
from utils import *

ensure_directory('dat')
ensure_directory('img')

parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="choose one of [minisat | cplex_rcw | cplex_region]")
parser.add_argument('seed', help="random seed", nargs='?', default=987, type=int)
parser.add_argument('delta', help="failure parameter", nargs='?', default=.01, type=float)
args = parser.parse_args()

np.random.seed(args.seed)

if args.dataset == "minisat":
    env = LBEnvironment('icar/dataset_icar/minisat_cnfuzzdd/measurements.dump', 900)
    num_phases = {'gam': 9, 'eps': 5, 'bal': 13, 'etg': 12}
elif args.dataset == "cplex_rcw":
    env = Environment('icar/dataset_icar/cplex_rcw/cplex_rcw_rt_seed{}.npy'.format(520), 10000)
    num_phases = {'gam': 15, 'eps': 7, 'bal': 12, 'etg': 9}
elif args.dataset == "cplex_region":
    env = Environment('icar/dataset_icar/cplex_region/cplex_region_rt_seed{}.npy'.format(520), 10000)
    num_phases = {'gam': 15, 'eps': 7, 'bal': 12, 'etg': 9}
else:
    print("\nERROR: dataset should be one of [minisat | cplex_rcw | cplex_region]\n")
    exit()

u = (u_ll, {'k0': 60, 'a': 1})
# u = (u_unif, {'k0': 60})
u_fn, u_params = u
u_str = u_to_str(u)

n_max = env.get_num_configs()
m_max = env.get_num_instances()


scenario = {
    'gam': {'epsilon_fn': lambda p: math.exp(- p / 30), 'gamma_fn': lambda p: math.exp(- p / 3), 'label': "$\\gamma$ focus", 'color': colors[3]},
    'etg': {'epsilon_fn': lambda p: math.exp(- p ** 3 / 300), 'gamma_fn': lambda p: math.exp(- p ** 2 / 30), 'label': "$\\gamma$ then $\\epsilon$", 'color': colors[5]},
    'bal': {'epsilon_fn': lambda p: math.exp(- p / 5), 'gamma_fn': lambda p: math.exp(- p / 5), 'label': "balanced", 'color': colors[7]},
    'eps': {'epsilon_fn': lambda p: math.exp(- p / 3), 'gamma_fn': lambda p: math.exp(- p / 30), 'label': "$\\epsilon$ focus", 'color': colors[2]},
    }

data = {}
for s, scen in scenario.items():
    data_save_path = "dat/explore_experiment_{}_{}_{}_seed={}.p".format(args.dataset, u_str, s, args.seed)
    print("Scenario {}, {} ...".format(s, scen['label']))
    try:
        data[s] = pickle.load(open(data_save_path, 'rb'))
        print("Loading existing data ...")
    except FileNotFoundError:
        data[s] = coup(env, lambda t: u_fn(t, **u_params), args.delta, scen['epsilon_fn'], scen['gamma_fn'], max_phases=num_phases[s], n_max=n_max, m_max=m_max, doubling_condition="new")
        pickle.dump(data[s], open(data_save_path, 'wb'))
        env.reset()
        

## epsilon vs gamma plot:
for s, scen in scenario.items():    
    epsilons = [data[s]['phase'][p]['epsilon'] for p in range(num_phases[s])]
    gammas = [scenario[s]['gamma_fn'](p + 1) for p in range(num_phases[s])]
    plt.plot(epsilons, gammas, c=scen['color'], marker='o', markersize=10, linewidth=lw['main'], label=scen['label'])
plt.xlim(1.005, -.005)
plt.ylim(1.005, -.005)
plt.xticks(fontsize=fs['ticks'])
plt.yticks(fontsize=fs['ticks'])
plt.legend(fontsize=fs['legend'], bbox_to_anchor=(1, .21), loc="lower right")
plt.xlabel("$\\epsilon$ guaranteed", fontsize=fs['axis'])
plt.ylabel("$\\gamma$ guaranteed", fontsize=fs['axis'])
plt.savefig("img/explore_experiment_eps_gam_{}_{}_seed={}.pdf".format(args.dataset, u_str, args.seed), bbox_inches='tight')
plt.clf()


## time vs eps:
for s, scen in scenario.items():    
    epsilons = [data[s]['phase'][p]['epsilon'] for p in range(num_phases[s])]
    times = [data[s]['phase'][p]['total_time'] for p in range(num_phases[s])]
    plt.plot(times, epsilons, c=scen['color'], marker='o', markersize=10, linewidth=lw['main'], label=scen['label'])
plt.ylim(-.005, 1.005)
plt.xticks(fontsize=fs['ticks'])
plt.yticks(fontsize=fs['ticks'])
plt.xscale('log')
plt.legend(fontsize=fs['legend'])
plt.xlabel("Total time (CPU days)", fontsize=fs['axis'])
plt.ylabel("$\\epsilon$ guaranteed", fontsize=fs['axis'])
plt.savefig("img/explore_experiment_time_eps_{}_{}_seed={}.pdf".format(args.dataset, u_str, args.seed), bbox_inches='tight')
plt.clf()


## time vs gam:
for s, scen in scenario.items():    
    gammas = [scenario[s]['gamma_fn'](p + 1) for p in range(num_phases[s])]
    times = [data[s]['phase'][p]['total_time'] for p in range(num_phases[s])]
    plt.plot(times, gammas, c=scen['color'], marker='o', markersize=10, linewidth=lw['main'], label=scen['label'])
plt.ylim(-.005, 1.005)
plt.xticks(fontsize=fs['ticks'])
plt.yticks(fontsize=fs['ticks'])
plt.xscale('log')
plt.legend(fontsize=fs['legend'])
plt.xlabel("Total time (CPU days)", fontsize=fs['axis'])
plt.ylabel("$\\gamma$ guaranteed", fontsize=fs['axis'])
plt.savefig("img/explore_experiment_time_gam_{}_{}_seed={}.pdf".format(args.dataset, u_str, args.seed), bbox_inches='tight')
plt.clf()




