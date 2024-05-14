import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

from environment import Environment, LBEnvironment, SyntheticEnvironment
from up import up
from naive import naive
from cuub import cuub, cuub_finite
from utils import *

ensure_directory('dat')
ensure_directory('img')

parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="choose one of [minisat | cplex_rcw | cplex_region]")
# parser.add_argument('num_phases', help="number of phases", nargs='?', default=11, type=int)
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

# u = (u_ll, {'k0': 120, 'a': 1})
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
    data['cuub'] = cuub(env, lambda t: u_fn(t, **u_params), args.delta, epsilon_fn, gamma_fn, max_phases=num_phases, n_max=n_max, m_max=m_max, doubling_condition="new")
    env.reset()

    data['finite'] = []
    for p in range(num_phases):
        num_configs = data['cuub']['phase'][p]['num_configs']
        epsilon = data['cuub']['phase'][p]['epsilon']
        d = cuub_finite(env, lambda t: u_fn(t, **u_params), args.delta, epsilon_min=epsilon, n=num_configs, m_max=m_max, doubling_condition="new")
        data['finite'].append(d)
        env.reset()

    print("Calculating mean utilities ...")
    u_vect = np.vectorize(lambda t: u_fn(t, **u_params), otypes=[float])
    data['utilities'] = [np.mean(u_vect(env._runtimes[i, :])) for i in range(env._num_configs)]

    pickle.dump(data, open(data_save_path, 'wb'))

total_time_per_phase_finite = [data['finite'][p]['total_times'][-1] * day_in_s for p in range(len(data['finite']))]
total_time_per_phase_cuub = [data['cuub']['phase'][p]['total_time'] * day_in_s for p in range(num_phases)]
epsilon_per_phase = [data['cuub']['phase'][p]['epsilon'] for p in range(num_phases)]

plt.plot(total_time_per_phase_finite, epsilon_per_phase, c=colors[2], marker='o', markersize=10, linewidth=lw['main'], label="FiniteCUUB")
plt.plot(total_time_per_phase_cuub, epsilon_per_phase, c=colors[3], marker='o', markersize=10, linewidth=lw['main'], label="CUUB")
plt.annotate("phase 1", (total_time_per_phase_cuub[0], epsilon_per_phase[0]))
for p in range(1, num_phases):
    plt.annotate(p+1, (total_time_per_phase_cuub[p], epsilon_per_phase[p]))
# plt.yticks([.05, .1, .2, .4], [".05", ".1", ".2", ".4"])
plt.ylim(0, 1)
plt.xscale('log')
# plt.yscale('log')
plt.legend()
plt.ylabel("$\\epsilon$ at end of phase", fontsize=fs['axis'])
# plt.xlabel("Total configuration time\nat end of phase (seconds)", fontsize=fs['axis'])
plt.xlabel("Total time (seconds)", fontsize=fs['axis'])
plt.title(args.dataset, fontsize=fs['title'])
plt.savefig("img/many_experiment_total_times_{}_{}_seed={}.pdf".format(args.dataset, u_str, args.seed), bbox_inches='tight')
plt.clf()

plt.scatter(data['utilities'], data['finite'][-1]['total_times_by_config'][-1], label="FiniteCUUB", c=colors[2])
plt.scatter(data['utilities'], data['cuub']['phase'][-1]['time_per_config'], label="CUUB", c=colors[3])
plt.yscale('log')
plt.legend()
plt.xlabel("Utility of configuration", fontsize=fs['axis'])
plt.ylabel("Time spent on configuration (seconds)", fontsize=fs['axis'])
plt.title("{}".format(args.dataset), fontsize=fs['title'])
plt.savefig("img/many_experiment_time_per_config_{}_{}_seed={}.pdf".format(args.dataset, u_str, args.seed), bbox_inches='tight')
plt.clf()

