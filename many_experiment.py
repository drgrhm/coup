import sys
import argparse
import math
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from environment import Environment, LBEnvironment, SyntheticEnvironment
from up import up
from naive import naive
from cuub import cuub, cuub_many
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="choose one of [minisat | cplex_rcw | cplex_region]")
parser.add_argument('seed', help="random seed", nargs='?', default=980, type=int)
parser.add_argument('delta', help="failure parameter", nargs='?', default=.01, type=float)
args = parser.parse_args()

np.random.seed(args.seed)

if args.dataset == "minisat":
    env = LBEnvironment('icar/dataset_icar/minisat_cnfuzzdd/measurements.dump', 900)
    pass
elif args.dataset == "cplex_rcw":
    env = Environment('icar/dataset_icar/cplex_rcw/cplex_rcw_rt_seed{}.npy'.format(520), 10000)
elif args.dataset == "cplex_region":
    env = Environment('icar/dataset_icar/cplex_region/cplex_region_rt_seed{}.npy'.format(520), 10000)
    pass
elif args.dataset == "synthetic":
    env = SyntheticEnvironment()
else:
    print("\nERROR: dataset should be one of [minisat | cplex_rcw | cplex_region]\n")
    exit()

u = (u_ll, {'k0': 60, 'a': 1})
# u = (u_unif, {'k0': 60})
# u = (u_ll, {'k0': 120, 'a': 2})
# u = (u_ll, {'k0': 240, 'a': .5})
# u = (u_exp, {'k0': 2000})
u_fn, u_params = u

if args.dataset == "synthetic":
    n_max = float('inf')
    m_max = float('inf')
else:
    n_max = env.get_num_configs()
    m_max = env.get_num_instances()

num_phases = 6

def gamma_fn(p):
    return 1 / 2 ** p

def epsilon_fn(p):
    return 1 / 2 ** p

data_save_path = "dat/many_experiment_{}_{}_seed={}.p".format(args.dataset, u_to_str(u), args.seed)

try: 
    (data_many, data_cuub) = pickle.load(open(data_save_path, 'rb'))
    print("Loading existing data ...")
except FileNotFoundError:
    
    data_many = cuub_many(env, lambda t: u_fn(t, **u_params), args.delta, epsilon_fn, gamma_fn, max_phases=num_phases, n_max=n_max, m_max=m_max)
    env.reset()

    data_cuub = []
    for pi, num_configs in enumerate(data_many['num_configs_per_phase']):
        d = cuub(env, lambda t: u_fn(t, **u_params), args.delta, epsilon_min=data_many['epsilon_at_end_of_phase'][pi], n=num_configs, m_max=m_max)
        data_cuub.append(d)
        env.reset()

    pickle.dump((data_many, data_cuub), open(data_save_path, 'wb'))

num_phases = len(data_many['total_time_at_end_of_phase'])

total_time_per_phase_cuub = [data_cuub[p]['total_times'][-1] * day_in_s for p in range(len(data_cuub))]
total_time_per_phase_many = [data_many['total_time_at_end_of_phase'][p] * day_in_s for p in range(num_phases)]
epsilon_per_phase = [data_many['epsilon_at_end_of_phase'][p] for p in range(num_phases)]

plt.plot(epsilon_per_phase, total_time_per_phase_cuub, c=colors[2], marker='o', markersize=num_phases, linewidth=lw['main'], label="CUUB")
plt.plot(epsilon_per_phase, total_time_per_phase_many, c=colors[1], marker='o', markersize=num_phases, linewidth=lw['main'], label="manyCUUB")

plt.annotate("phase 1", (epsilon_per_phase[0], total_time_per_phase_many[0]))
for p in range(1, num_phases):
    plt.annotate(p+1, (epsilon_per_phase[p], total_time_per_phase_many[p]))

plt.xscale('log')
plt.yscale('log')
if args.dataset == 'synthetic':
    plt.xticks([.0125, .025, .05, .1, .2, .4], [".0125", ".025", ".05", ".1", ".2", ".4"])
else:
    plt.xticks([.05, .1, .2, .4], [".05", ".1", ".2", ".4"])
# plt.xlim(.0125, 1)
# plt.ylim(2e3, 2e8)
plt.xlabel("$\epsilon$ at end of phase", fontsize=fs['axis'])
plt.ylabel("total configuration time\nat end of phase (seconds)", fontsize=fs['axis'])
plt.title(args.dataset, fontsize=fs['title'])
plt.legend()
plt.savefig("img/many_experiment_total_times_{}_{}_seed={}.pdf".format(args.dataset, u_to_str(u), args.seed), bbox_inches='tight')
plt.clf()





