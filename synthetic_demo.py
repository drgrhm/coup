import sys
import argparse
import math
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from environment import Environment, LBEnvironment, SyntheticEnvironment
from utils import *



def plot_tartans(seed=None, cmap='viridis_r'):

    print("Running plot_tartans, seed={}".format(seed))

    np.random.seed(seed)

    env_sat = LBEnvironment('icar/dataset_icar/minisat_cnfuzzdd/measurements.dump', 900)
    env_rcw = Environment('icar/dataset_icar/cplex_rcw/cplex_rcw_rt_seed{}.npy'.format(520), 10000)
    env_reg = Environment('icar/dataset_icar/cplex_region/cplex_region_rt_seed{}.npy'.format(520), 10000)
    env_sim = SyntheticEnvironment()

    num_configs = 201
    num_instances = 801
    kmax = 10000 # max captime of cplex data 

    runtimes_sat = np.zeros((num_configs, num_instances))
    runtimes_rcw = np.zeros((num_configs, num_instances))
    runtimes_reg = np.zeros((num_configs, num_instances))
    runtimes_sim = np.zeros((num_configs, num_instances))

    logruntimes_sat = np.zeros((num_configs, num_instances))
    logruntimes_rcw = np.zeros((num_configs, num_instances))
    logruntimes_reg = np.zeros((num_configs, num_instances))
    logruntimes_sim = np.zeros((num_configs, num_instances))

    configs_sat = np.random.choice(env_sat.get_num_configs(), size=num_configs, replace=False)
    instances_sat = np.random.choice(env_sat.get_num_instances(), size=num_instances, replace=False)

    configs_rcw = np.random.choice(env_rcw.get_num_configs(), size=num_configs, replace=False)
    instances_rcw = np.random.choice(env_rcw.get_num_instances(), size=num_instances, replace=False)

    configs_reg = np.random.choice(env_reg.get_num_configs(), size=num_configs, replace=False)
    instances_reg = np.random.choice(env_reg.get_num_instances(), size=num_instances, replace=False)

    for i in range(num_configs):
        for j in range(num_instances):
            runtimes_sat[i, j] = env_sat._runtimes[configs_sat[i], instances_sat[j]] * kmax / 900 # scale to range of other datasets
            runtimes_rcw[i, j] = env_rcw._runtimes[configs_rcw[i], instances_rcw[j]]
            runtimes_reg[i, j] = env_reg._runtimes[configs_reg[i], instances_reg[j]]
            runtimes_sim[i, j] = env_sim.run(i, j, kmax)

            logruntimes_sat[i, j] = math.log(runtimes_sat[i, j])
            logruntimes_rcw[i, j] = math.log(runtimes_rcw[i, j])
            logruntimes_reg[i, j] = math.log(runtimes_reg[i, j])
            logruntimes_sim[i, j] = math.log(runtimes_sim[i, j])

    ## linear plot
    fig, axs = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
    axs[0].imshow(runtimes_sat, cmap=cmap)
    axs[1].imshow(runtimes_rcw, cmap=cmap)
    axs[2].imshow(runtimes_reg, cmap=cmap)
    pcm = axs[3].imshow(runtimes_sim, cmap=cmap)
    # cbar = fig.colorbar(pcm, ax=axs, shrink=.9, label="runtime")
    cbar = fig.colorbar(pcm, ax=axs, shrink=.9)
    cbar.set_label(label="runtime", size=fs['axis'])
    # cbar.ax.set_fontsize(fs['axis'])

    for i in range(4):
        axs[i].yaxis.set_label_position("right")
    axs[0].set_ylabel("minisat$^*$", fontsize=fs['axis'])
    axs[1].set_ylabel("cplex_rcw", fontsize=fs['axis'])
    axs[2].set_ylabel("cplex_region", fontsize=fs['axis'])
    axs[3].set_ylabel("synthetic", fontsize=fs['axis'])

    plt.xlabel("Instance", fontsize=fs['title'])
    fig.text(0.04, 0.5, 'Configuration', va='center', rotation='vertical', fontsize=fs['title'])
    plt.savefig("img/data_tartan_seed={}.pdf".format(seed), bbox_inches='tight', dpi=1000)
    plt.clf()

    ## log plot
    fig, axs = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
    axs[0].imshow(logruntimes_sat, cmap=cmap)
    axs[1].imshow(logruntimes_rcw, cmap=cmap)
    axs[2].imshow(logruntimes_reg, cmap=cmap)
    pcm = axs[3].imshow(logruntimes_sim, cmap=cmap)
    # cbar = fig.colorbar(pcm, ax=axs, shrink=.9, label="log(runtime)")
    cbar = fig.colorbar(pcm, ax=axs, shrink=.9)
    cbar.set_label(label="log(runtime)", size=fs['axis'])
    # cbar.ax.set_fontsize(fs['axis'])

    for i in range(4):
        axs[i].yaxis.set_label_position("right")
    axs[0].set_ylabel("minisat$^*$", fontsize=fs['axis'])
    axs[1].set_ylabel("cplex_rcw", fontsize=fs['axis'])
    axs[2].set_ylabel("cplex_region", fontsize=fs['axis'])
    axs[3].set_ylabel("synthetic", fontsize=fs['axis'])

    plt.xlabel("Instance", fontsize=fs['title'])
    fig.text(0.04, 0.5, 'Configuration', va='center', rotation='vertical', fontsize=fs['title'])
    plt.savefig("img/data_tartan_log_seed={}.pdf".format(seed), bbox_inches='tight', dpi=1000)
    plt.clf()



def plot_comparative_cdfs(seed=None, log=False):

    print("Running plot_comparative_cdfs, seed={}, log={}".format(seed, log))

    np.random.seed(seed)

    env_sat = LBEnvironment('icar/dataset_icar/minisat_cnfuzzdd/measurements.dump', 900)
    env_rcw = Environment('icar/dataset_icar/cplex_rcw/cplex_rcw_rt_seed{}.npy'.format(520), 10000)
    env_reg = Environment('icar/dataset_icar/cplex_region/cplex_region_rt_seed{}.npy'.format(520), 10000)
    env_sim = SyntheticEnvironment()

    ls = ':'

    num_configs_sim = 200
    num_instances_sim = 2000 
    kmax = 10000 # max captime of cplex data 

    runtimes_sim = np.zeros((num_configs_sim, num_instances_sim))
    for i in range(num_configs_sim):
        for j in range(num_instances_sim):
            runtimes_sim[i, j] = env_sim.run(i, j, kmax)

    runtimes_sat = env_sat._runtimes * kmax / 900 # scale to range of other datasets
    runtimes_rcw = env_rcw._runtimes
    runtimes_reg = env_reg._runtimes

    if log:
        print("Converting to log runtimes...")
        runtimes_sat = np.log(env_sat._runtimes)
        runtimes_rcw = np.log(env_rcw._runtimes)
        runtimes_reg = np.log(env_reg._runtimes)
        runtimes_sim = np.log(runtimes_sim)

    mean_configs_sat = np.mean(runtimes_sat, 1)
    mean_configs_rcw = np.mean(runtimes_rcw, 1)
    mean_configs_reg = np.mean(runtimes_reg, 1)
    mean_configs_sim = np.mean(runtimes_sim, 1)

    plt.plot(*ecdf(mean_configs_sat), label="minisat", linestyle=ls, linewidth=lw['main'], color=colors[0])
    plt.plot(*ecdf(mean_configs_rcw), label="cplex_rcw", linestyle=ls, linewidth=lw['main'], color=colors[1])
    plt.plot(*ecdf(mean_configs_reg), label="cplex_region", linestyle=ls, linewidth=lw['main'], color=colors[2])
    plt.plot(*ecdf(mean_configs_sim), label="synthetic", linewidth=lw['main'], color=colors[3])
    plt.title("config", fontsize=fs['title'])
    plt.legend()
    if log:        
        plt.xlabel("log(t)", fontsize=fs['axis'])
        plt.ylabel("Pr(log(runtime) < log(t))", fontsize=fs['axis'])
        plt.savefig("img/cdf_compare_config_log_seed={}.pdf".format(seed), bbox_inches='tight')
    else:        
        plt.xlabel("t", fontsize=fs['axis'])
        plt.ylabel("Pr(runtime < t)", fontsize=fs['axis'])
        plt.savefig("img/cdf_compare_config_seed={}.pdf".format(seed), bbox_inches='tight')
    plt.clf()

    mean_instances_sat = np.mean(runtimes_sat, 0)
    mean_instances_rcw = np.mean(runtimes_rcw, 0)
    mean_instances_reg = np.mean(runtimes_reg, 0)
    mean_instances_sim = np.mean(runtimes_sim, 0)    

    plt.plot(*ecdf(mean_instances_sat), label="minisat", linestyle=ls, linewidth=lw['main'], color=colors[0])
    plt.plot(*ecdf(mean_instances_rcw), label="cplex_rcw", linestyle=ls, linewidth=lw['main'], color=colors[1])
    plt.plot(*ecdf(mean_instances_reg), label="cplex_region", linestyle=ls, linewidth=lw['main'], color=colors[2])
    plt.plot(*ecdf(mean_instances_sim), label="synthetic", linewidth=lw['main'], color=colors[3])
    plt.title("instance", fontsize=fs['title'])
    plt.legend()
    if log:                
        plt.xlabel("log(t)", fontsize=fs['axis'])
        plt.ylabel("Pr(log(runtime) < log(t))", fontsize=fs['axis'])
        plt.savefig("img/cdf_compare_instance_log_seed={}.pdf".format(seed), bbox_inches='tight')
    else:        
        plt.xlabel("t", fontsize=fs['axis'])
        plt.ylabel("Pr(runtime < t)", fontsize=fs['axis'])
        plt.savefig("img/cdf_compare_instance_seed={}.pdf".format(seed), bbox_inches='tight')
    plt.clf()

    n_plot = 100000
    print("Sampling flattened runtimes...")
    runtimes_plot_sat = np.random.choice(runtimes_sat.flatten(), size=n_plot, replace=False)
    runtimes_plot_rcw = np.random.choice(runtimes_rcw.flatten(), size=n_plot, replace=False)
    runtimes_plot_reg = np.random.choice(runtimes_reg.flatten(), size=n_plot, replace=False)
    runtimes_plot_sim = np.random.choice(runtimes_sim.flatten(), size=n_plot, replace=False)

    plt.plot(*ecdf(runtimes_plot_sat), label="minisat", linestyle=ls, linewidth=lw['main'], color=colors[0])
    plt.plot(*ecdf(runtimes_plot_rcw), label="cplex_rcw", linestyle=ls, linewidth=lw['main'], color=colors[1])
    plt.plot(*ecdf(runtimes_plot_reg), label="cplex_region", linestyle=ls, linewidth=lw['main'], color=colors[2])
    plt.plot(*ecdf(runtimes_plot_sim), label="synthetic", linewidth=lw['main'], color=colors[3])
    plt.title("flattened", fontsize=fs['title'])
    plt.legend()
    if log:        
        plt.xlabel("log(t)", fontsize=fs['axis'])
        plt.ylabel("Pr(log(runtime) < log(t))", fontsize=fs['axis'])
        plt.savefig("img/cdf_compare_flat_log_seed={}.pdf".format(seed), bbox_inches='tight')
    else:        
        plt.xlabel("t", fontsize=fs['axis'])
        plt.ylabel("Pr(runtime < t)", fontsize=fs['axis'])
        plt.savefig("img/cdf_compare_flat_seed={}.pdf".format(seed), bbox_inches='tight')
    plt.clf()

    num_configs_plot = 5
    configs_sat = np.random.choice(env_sat.get_num_configs(), size=num_configs_plot, replace=False)
    configs_rcw = np.random.choice(env_rcw.get_num_configs(), size=num_configs_plot, replace=False)
    configs_reg = np.random.choice(env_reg.get_num_configs(), size=num_configs_plot, replace=False)
    configs_sim = np.random.choice(num_configs_sim, size=num_configs_plot, replace=False)

    for i in range(num_configs_plot):
        plt.plot(*ecdf(runtimes_sat[configs_sat[i], :]), linewidth=lw['tiny'], color=colors[0])
        plt.plot(*ecdf(runtimes_rcw[configs_rcw[i], :]), linewidth=lw['tiny'], color=colors[1])
        plt.plot(*ecdf(runtimes_reg[configs_reg[i], :]), linewidth=lw['tiny'], color=colors[2])
        plt.plot(*ecdf(runtimes_sim[configs_sim[i], :]), linewidth=lw['small'], color=colors[3])

    if log:
        plt.legend(["minisat", "cplex_rcw", "cplex_region", "synthetic"], loc='upper left')
        plt.xlabel("log(t)", fontsize=fs['axis'])
        plt.ylabel("Pr(log(runtime) < log(t))", fontsize=fs['axis'])
        plt.savefig("img/cdf_compare_by_config_log_seed={}.pdf".format(seed), bbox_inches='tight')
    else:
        plt.legend(["minisat", "cplex_rcw", "cplex_region", "synthetic"], loc='lower right')
        plt.xlabel("t", fontsize=fs['axis'])
        plt.ylabel("Pr(runtime < t)", fontsize=fs['axis'])
        plt.savefig("img/cdf_compare_by_config_seed={}.pdf".format(seed), bbox_inches='tight')
    plt.clf()



if __name__ == "__main__":

    np.set_printoptions(linewidth=300)

    global_seed = 985
    np.random.seed(global_seed)
    seeds = np.random.choice(1000, size=1, replace=False)

    for seed in seeds:
        print("Running seed={}".format(seed))
        plot_tartans(seed=seed)
        plot_comparative_cdfs(seed=seed, log=True)








