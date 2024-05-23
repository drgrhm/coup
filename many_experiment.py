import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

from environment import Environment, LBEnvironment
from coup import oup, coup
from utils import *

ensure_directory('dat')
ensure_directory('img')

parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="choose one of [minisat | cplex_rcw | cplex_region]")
parser.add_argument('seed', help="random seed", nargs='?', default=985, type=int)
parser.add_argument('delta', help="failure parameter", nargs='?', default=.01, type=float)
args = parser.parse_args()

np.random.seed(args.seed)

if args.dataset == "minisat":
    env = LBEnvironment('icar/dataset_icar/minisat_cnfuzzdd/measurements.dump', 900)
    num_phases = 13
elif args.dataset == "cplex_rcw":
    env = Environment('icar/dataset_icar/cplex_rcw/cplex_rcw_rt_seed{}.npy'.format(520), 10000)
    num_phases = 15
elif args.dataset == "cplex_region":
    env = Environment('icar/dataset_icar/cplex_region/cplex_region_rt_seed{}.npy'.format(520), 10000)
    num_phases = 15
else:
    print("\nERROR: dataset should be one of [minisat | cplex_rcw | cplex_region]\n")
    exit()

u = (u_ll, {'k0': 60, 'a': 1})
# u = (u_unif, {'k0': 60})
u_fn, u_params = u
u_str = u_to_str(u)

n_max = env.get_num_configs()
m_max = env.get_num_instances()

def epsilon_fn(p):
    return math.exp(- (p / 6))


def gamma_fn(p):
    return math.exp(- p / 3)


data_save_path = "dat/many_experiment_{}_{}_seed={}.p".format(args.dataset, u_str, args.seed)

try: 
    data = pickle.load(open(data_save_path, 'rb'))
    print("Loading existing data ...")
except FileNotFoundError:
    data = {}
    data['coup'] = coup(env, lambda t: u_fn(t, **u_params), args.delta, epsilon_fn, gamma_fn, max_phases=num_phases, n_max=n_max, m_max=m_max, doubling_condition="new")
    env.reset()

    data['oup'] = []
    for p in range(num_phases):
        num_configs = data['coup']['phase'][p]['num_configs']
        epsilon = data['coup']['phase'][p]['epsilon']
        d = oup(env, lambda t: u_fn(t, **u_params), args.delta, epsilon_min=epsilon, n=num_configs, m_max=m_max, doubling_condition="new")
        data['oup'].append(d)
        env.reset()

    print("Calculating mean utilities ...")
    u_vect = np.vectorize(lambda t: u_fn(t, **u_params), otypes=[float])
    data['utilities'] = np.mean(u_vect(env._runtimes), 1)

    pickle.dump(data, open(data_save_path, 'wb'))

total_time_per_phase_oup = [data['oup'][p]['total_times'][-1] for p in range(len(data['oup']))]
total_time_per_phase_coup = [data['coup']['phase'][p]['total_time'] for p in range(num_phases)]
epsilon_per_phase = [data['coup']['phase'][p]['epsilon'] for p in range(num_phases)]

plt.plot(total_time_per_phase_oup, epsilon_per_phase, c=colors[2], marker='o', markersize=10, linewidth=lw['main'], label="OUP")
plt.plot(total_time_per_phase_coup, epsilon_per_phase, c=colors[3], marker='o', markersize=10, linewidth=lw['main'], label="COUP")
plt.annotate("phase 1", (total_time_per_phase_coup[0], epsilon_per_phase[0]))
for p in range(1, num_phases):
    plt.annotate(p+1, (total_time_per_phase_coup[p], epsilon_per_phase[p]))
plt.xticks(fontsize=fs['ticks'])
plt.yticks(fontsize=fs['ticks'])
plt.ylim(0, 1)
plt.xscale('log')
plt.legend(fontsize=fs['legend'])
plt.ylabel("$\\epsilon$ at end of phase", fontsize=fs['axis'])
plt.xlabel("Total time (CPU days)", fontsize=fs['axis'])
plt.title(args.dataset, fontsize=fs['title'])
plt.savefig("img/many_experiment_total_times_{}_{}_seed={}.pdf".format(args.dataset, u_str, args.seed), bbox_inches='tight')
plt.clf()

plt.scatter(data['utilities'], data['oup'][-1]['total_times_by_config'][-1], label="OUP", c=colors[2])
plt.scatter(data['utilities'], data['coup']['phase'][-1]['time_per_config'], label="COUP", c=colors[3])
# plt.ylim(9e2, 5e6)
plt.xticks(fontsize=fs['ticks'])
plt.yticks(fontsize=fs['ticks'])
plt.locator_params(axis='x', nbins=6)
plt.yscale('log')
plt.legend(fontsize=fs['legend'], loc='upper left')
plt.xlabel("Utility of configuration", fontsize=fs['axis'])
plt.ylabel("Time per configuration (s)", fontsize=fs['axis'])
plt.title("{}".format(args.dataset), fontsize=fs['title'])
plt.savefig("img/many_experiment_time_per_config_{}_{}_seed={}.pdf".format(args.dataset, u_str, args.seed), bbox_inches='tight')
plt.clf()

