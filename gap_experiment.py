import math
import argparse
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, LogLocator)

from environment import Environment, LBEnvironment
from up import up
from coup import oup, coup, coupnm, coupnmbest
from successive_halving import sh
from hyperband import hyperband
from structured_procrastination import spc

from impatient import ImpatientCapsAndRuns

from utils import ensure_directory, u_ll, u_unif, u_to_str, day_in_s, lw, fs, colors, color_schemes

from ac_band import ac_band

from minorsymloglocator import MinorSymLogLocator


def load_env(dataset, seed=520):
    if dataset == "minisat":
        np.random.seed(seed)
        env = LBEnvironment('icar/dataset_icar/minisat_cnfuzzdd/measurements.dump', 900)
    elif dataset == "cplex_rcw":
        env = Environment('icar/dataset_icar/cplex_rcw/cplex_rcw_rt_seed{}.npy'.format(seed), 10000)
    elif dataset == "cplex_region":
        env = Environment('icar/dataset_icar/cplex_region/cplex_region_rt_seed{}.npy'.format(seed), 10000)
    else:
        print("\nERROR: dataset should be one of [minisat | cplex_rcw | cplex_region]\n")
        exit()
    return env


def create_utility_plot_array(data, alg_key, seeds, max_time, xs):
    Y = np.ones((len(seeds), xs.shape[0]))
    for ii, seed in enumerate(seeds):
        for xi, x in enumerate(xs):
            # the right Y value is the biggest one that is still smaller than x:
            total_times = np.array(data[seed][alg_key]['total_time'])
            total_times[total_times >= x] = -1
            if np.max(total_times) == -1:
                Y[ii, xi] = 1
            else:
                i_star = data[seed][alg_key]['i_stars'][np.argmax(total_times)]
                Y[ii, xi] = data[seed]['utilities'][i_star]
    optimals = np.array([np.max(data[seed]['utilities']) for seed in seeds])[:, np.newaxis]
    Y = 1 - Y / optimals
    return Y


def create_runtime_plot_array(data, alg_key, seeds, max_time, xs, delta_i):
    Y = np.ones((len(seeds), xs.shape[0]))
    for ii, seed in enumerate(seeds):
        for xi, x in enumerate(xs):
            # the right Y value is the biggest one that is still smaller than x:
            total_times = np.array(data[seed][alg_key]['total_time'])
            total_times[total_times >= x] = -1
            if np.max(total_times) == -1:
                Y[ii, xi] = 1
            else:
                i_star = data[seed][alg_key]['i_stars'][np.argmax(total_times)]
                # Y[ii, xi] = data[seed]['utilities'][i_star]
                Y[ii, xi] = data[seed]['mean_runtimes'][delta_i][i_star]

    # optimals = np.array([np.max(data[seed]['utilities']) for seed in seeds])[:, np.newaxis]
    optimals = np.array([np.min(data[seed]['mean_runtimes'][delta_i]) for seed in seeds])[:, np.newaxis]
    Y = 1 - optimals / Y
    return Y



def draw_plot_curve(data, alg_key, color, seeds, is_runtime=False, delta_i=None, num_pts=50000, zorder=None):
    max_time = min([data[seed][alg_key]['total_time'][-1] for seed in seeds]) * day_in_s
    xs = np.linspace(0, max_time / day_in_s, num_pts)
    if is_runtime:
        Y = create_runtime_plot_array(data, alg_key, seeds, max_time, xs, delta_i)
    else:
        Y = create_utility_plot_array(data, alg_key, seeds, max_time, xs)
    plt.plot(xs, np.mean(Y, axis=0), c=color, linewidth=lw['main'], zorder=zorder, label=plot_labels[alg_key])
    # plt.plot(xs, np.min(Y, axis=0), c=color, linewidth=.5, zorder=zorder)
    # plt.plot(xs, np.max(Y, axis=0), c=color, linewidth=.5, zorder=zorder)
    plt.fill_between(xs, np.min(Y, axis=0), np.max(Y, axis=0), color=color, alpha=.15)


def plot_rectangle(xs, ys, color, alpha=.1, linewidth=0.1, label=None, zorder=None):
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    plt.fill_between([min_x, max_x], max_y, min_y, color=color, alpha=alpha, zorder=zorder)
    # plt.plot([min_x, max_x], [min_y, min_y], color=color, linewidth=linewidth, zorder=zorder)
    # plt.plot([min_x, min_x], [min_y, max_y], color=color, linewidth=linewidth, zorder=zorder)
    # plt.plot([max_x, max_x], [max_y, min_y], color=color, linewidth=linewidth, zorder=zorder)
    # plt.plot([max_x, min_x], [max_y, max_y], color=color, linewidth=linewidth, zorder=zorder)
    plt.scatter(np.mean(xs), np.mean(ys), color=color, zorder=zorder, label=label, marker="x", s=130)



def epsilon_fn(p):
    return math.exp(- (p / 6))

def gamma_fn(p):
    return math.exp(- p / 3)


utility_function_by_name = {
    'u_unif': u_unif,
    'u_ll': u_ll,
}


def parse_u(s):
    """ for parsing utility functions from command line arguments """
    ss = s.split(" ")
    u_fn = utility_function_by_name[ss[0]]
    u_params = {}
    for p in ss[1:]:
        pname, pval = p.split("=")
        u_params[pname] = float(pval)
    expected_params = [p for p in u_fn.__code__.co_varnames if p != 't']
    for p in expected_params:
        if p not in u_params:
            print(f"utility function {ss[0]} requires parameters {expected_params} but is missing {p}.")
            exit()
    return (u_fn, u_params)


if __name__ == "__main__":

    ensure_directory('dat')
    ensure_directory('img/log')
    ensure_directory('logs/acband')

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="choose one of [minisat | cplex_rcw | cplex_region]")    
    parser.add_argument('--delta', help="failure probability", nargs='?', default=0.05, type=float)
    parser.add_argument('--numphases', help="number of phases to run coup", nargs='?', default=20, type=int)
    parser.add_argument('--u', help="utility function", nargs='?', default="u_ll k0=60 a=1", type=parse_u)
    parser.add_argument('--seed', help="random seed", nargs='?', default=985, type=int)
    args = parser.parse_args()


    u_fn, u_params = args.u
    u_str = u_to_str(args.u)


    ## AC-Band Params:
    acb_ks = [2, 4, 8]
    acb_alphas = [0.05, 0.02, 0.01]

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger_path = f"./logs/acband/AC-Band.log"
    fileh = logging.FileHandler(logger_path)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    fileh.setFormatter(formatter)
    log = logging.getLogger()
    for hdlr in log.handlers[:]:
        log.removeHandler(hdlr)
    log.addHandler(fileh)
    log.addHandler(logging.StreamHandler())


    ## SH params:
    if args.dataset == 'minisat':
        Bs = [22000, 104000, 186000] # B=22000 is about min possible for cplex_rcw, B=350000 is about max
        ks_sh = [1, 10, 100, 500]
    else:
        Bs = [22000, 104000, 186000, 268000, 350000] # B=22000 is about min possible for cplex_rcw, B=350000 is about max
        ks_sh = [10, 100, 1000, 5000]

    ## Hyperband params:
    if args.dataset == 'minisat':
        etas_and_Rs = [(2, 500), (3, 700), (4, 1000), (5, 3100), (6, 1200)]
        ks_hb = [1, 10, 100, 500]
    else:
        etas_and_Rs = [(2, 1000), (3, 2000), (4, 4000), (5, 3000), (6, 3000)]
        ks_hb = [10, 100, 1000, 5000]


    ## ICAR params:
    deltas = [0.1, 0.2]
    if args.dataset == 'minisat':        
        epsilons = [.25]
    else:
        epsilons = [.1]
    gammas = [0.01, 0.02, 0.05]
    configs = [
        (False, False, True),
        (False, False, False),
        (True, True, True),
        (True, False, True)
    ]


    ### SPC params:
    if args.dataset == 'minisat':
        stop_time = 500 # in CPU days
    elif args.dataset == 'cplex_region':
        stop_time = 1600 
    else:
        stop_time = 5000 
    stop_time = stop_time * day_in_s # in seconds


    plot_labels = {'coup': "COUPeg", 'oup': "OUP", 'spc': "SPC", 'coupnm': "COUP"}

    cmap = plt.cm.tab10
    plot_colors = {
        'coup': colors[3], 
        'oup': colors[2], 
        'spc': cmap(0),
        'icar': cmap(8),
        'sh': cmap(4),
        'hb': cmap(9),
        'acb': cmap(3),
        'acbu': cmap(1),
        }

    seeds = [520 + s for s in range(5)]
    data = {}

    for seed in seeds:
        data[seed] = {'acb9': {}, 'acb10': {}, 'acbu9': {}, 'acbu10': {}}
        print("Creating environment for {}, seed={}".format(args.dataset, seed), flush=True)
        np.random.seed(args.seed)
        env = load_env(args.dataset, seed)
        np.random.seed(seed)
        for k in acb_ks:
            data[seed]['acb9'][k] = {}
            data[seed]['acb10'][k] = {}
            data[seed]['acbu9'][k] = {}
            data[seed]['acbu10'][k] = {}
            for ai, alpha in enumerate(acb_alphas):

                data_save_path = "dat/gap_experiment_{}_{}_{}_seed={}_acb900_k={}_alpha={}.p".format(args.dataset, args.seed, u_str, seed, k, alpha)
                try: 
                    data[seed]['acb9'][k][ai] = pickle.load(open(data_save_path, 'rb'))
                    print("Loading existing data for AC-Band, k={} max_cap={} ...".format(k, 900), flush=True)
                except FileNotFoundError:
                    np.random.seed(seed)
                    out = ac_band(env, list(range(env.num_configs)), env.num_instances, env.num_instances, k, alpha,  args.delta, max_cap=900, u=None)
                    data[seed]['acb9'][k][ai] = list(out)
                    data[seed]['acb9'][k][ai].append(env.total_times)
                    env.reset()
                    pickle.dump(data[seed]['acb9'][k][ai], open(data_save_path, 'wb'))


                if args.dataset != 'minisat':
                    data_save_path = "dat/gap_experiment_{}_{}_{}_seed={}_acb10000_k={}_alpha={}.p".format(args.dataset, args.seed, u_str, seed, k, alpha)
                    try: 
                        data[seed]['acb10'][k][ai] = pickle.load(open(data_save_path, 'rb'))
                        print("Loading existing data for AC-Band, k={}, max_cap={} ...".format(k, 10000), flush=True)
                    except FileNotFoundError:
                        np.random.seed(seed)
                        out = ac_band(env, list(range(env.num_configs)), env.num_instances, env.num_instances, k, alpha,  args.delta, max_cap=10000, u=None)
                        data[seed]['acb10'][k][ai] = list(out)
                        data[seed]['acb10'][k][ai].append(env.total_times)                
                        env.reset()
                        pickle.dump(data[seed]['acb10'][k][ai], open(data_save_path, 'wb'))


                data_save_path = "dat/gap_experiment_{}_{}_{}_seed={}_acbu900_k={}_alpha={}.p".format(args.dataset, args.seed, u_str, seed, k, alpha)
                try: 
                    data[seed]['acbu9'][k][ai] = pickle.load(open(data_save_path, 'rb'))
                    print("Loading existing data for AC-Band optimizing u={}, k={}, max_cap={} ...".format(u_str, k, 900), flush=True)
                except FileNotFoundError:
                    np.random.seed(seed)
                    out = ac_band(env, list(range(env.num_configs)), env.num_instances, env.num_instances, k, alpha,  args.delta, max_cap=900, u=lambda t: u_fn(t, **u_params))
                    data[seed]['acbu9'][k][ai] = list(out)
                    data[seed]['acbu9'][k][ai].append(env.total_times)
                    env.reset()
                    pickle.dump(data[seed]['acbu9'][k][ai], open(data_save_path, 'wb'))


                if args.dataset != 'minisat':
                    data_save_path = "dat/gap_experiment_{}_{}_{}_seed={}_acbu10000_k={}_alpha={}.p".format(args.dataset, args.seed, u_str, seed, k, alpha)
                    try: 
                        data[seed]['acbu10'][k][ai] = pickle.load(open(data_save_path, 'rb'))
                        print("Loading existing data for AC-Band optimizing u={}, k={}, max_cap={} ...".format(u_str, k, 10000), flush=True)
                    except FileNotFoundError:
                        np.random.seed(seed)
                        out = ac_band(env, list(range(env.num_configs)), env.num_instances, env.num_instances, k, alpha,  args.delta, max_cap=10000, u=lambda t: u_fn(t, **u_params))
                        data[seed]['acbu10'][k][ai] = list(out)
                        data[seed]['acbu10'][k][ai].append(env.total_times)                
                        env.reset()
                        pickle.dump(data[seed]['acbu10'][k][ai], open(data_save_path, 'wb'))


        data_save_path = "dat/gap_experiment_{}_{}_{}_seed={}_coup.p".format(args.dataset, args.seed, u_str, seed)
        try: 
            data[seed]['coup'] = pickle.load(open(data_save_path, 'rb'))
            print("Loading existing data for COUP ...", flush=True)
        except FileNotFoundError:
            np.random.seed(seed)
            data[seed]['coup'] = coup(env, lambda t: u_fn(t, **u_params), args.delta, epsilon_fn, gamma_fn, max_phases=args.numphases, n_max=env.num_configs, m_max=env.num_instances, doubling_condition="new", improved_tie_breaking=True)
            env.reset()
            pickle.dump(data[seed]['coup'], open(data_save_path, 'wb'))


        data_save_path = "dat/gap_experiment_{}_{}_{}_seed={}_coupnm.p".format(args.dataset, args.seed, u_str, seed)
        try: 
            data[seed]['coupnm'] = pickle.load(open(data_save_path, 'rb'))
            print("Loading existing data for COUP ...", flush=True)
        except FileNotFoundError:
            np.random.seed(seed)
            num_phases = 14            
            ms = list(reversed([int(env.num_instances / 1.5**r) for r in range(num_phases)]))
            ns = list(reversed([int(env.num_configs / 1.5**r) for r in range(num_phases)]))
            data[seed]['coupnm'] = coupnm(env, lambda t: u_fn(t, **u_params), args.delta, ns, ms, max_phases=args.numphases, n_max=env.num_configs, m_max=env.num_instances, doubling_condition="new", improved_tie_breaking=True)
            env.reset()
            pickle.dump(data[seed]['coupnm'], open(data_save_path, 'wb'))


        data_save_path = "dat/gap_experiment_{}_{}_{}_seed={}_oup.p".format(args.dataset, args.seed, u_str, seed)
        try: 
            data[seed]['oup'] = pickle.load(open(data_save_path, 'rb'))
            print("Loading existing data for OUP ...", flush=True)
        except FileNotFoundError:
            np.random.seed(seed)
            data[seed]['oup'] = oup(env, lambda t: u_fn(t, **u_params), args.delta, m_max=env.num_instances, improved_tie_breaking=True)
            env.reset()
            pickle.dump(data[seed]['oup'], open(data_save_path, 'wb'))


        data_save_path = "dat/gap_experiment_{}_{}_{}_seed={}_sh.p".format(args.dataset, args.seed, u_str, seed)
        try: 
            data[seed]['sh'] = pickle.load(open(data_save_path, 'rb'))
            print("Loading existing data for SH ...", flush=True)
        except FileNotFoundError:
            np.random.seed(seed)
            data[seed]['sh'] = {}
            for B in Bs:
                data[seed]['sh'][B] = {}
                for k in ks_sh:
                    print("Running SH with B={}, k={}".format(B, k), flush=True)
                    data[seed]['sh'][B][k] = sh(env, lambda t: u_fn(t, **u_params), k, B)
                    env.reset()
            pickle.dump(data[seed]['sh'], open(data_save_path, 'wb'))


        data_save_path = "dat/gap_experiment_{}_{}_{}_seed={}_hb.p".format(args.dataset, args.seed, u_str, seed)
        try: 
            data[seed]['hb'] = pickle.load(open(data_save_path, 'rb'))
            print("Loading existing data for Hyperband ...", flush=True)
        except FileNotFoundError:
            np.random.seed(seed)
            data[seed]['hb'] = {}
            for eta, R in etas_and_Rs:
                data[seed]['hb'][R] = {}
                for k in ks_hb:
                    print("Running Hyperband with R={}, k={}".format(R, k), flush=True)
                    data[seed]['hb'][R][k] = hyperband(env, lambda t: u_fn(t, **u_params), k, R=R, eta=eta)
                    env.reset()
            pickle.dump(data[seed]['hb'], open(data_save_path, 'wb'))


        data_save_path = "dat/gap_experiment_{}_{}_{}_seed={}_spc.p".format(args.dataset, args.seed, u_str, seed)
        try: 
            data[seed]['spc'] = pickle.load(open(data_save_path, 'rb'))
            print("Loading existing data for SPC ...", flush=True)
        except FileNotFoundError:
            np.random.seed(seed)
            print("Running SPC ...", flush=True)
            data[seed]['spc'] = spc(env, 1, 2, stop_time)
            env.reset()
            pickle.dump(data[seed]['spc'], open(data_save_path, 'wb'))


        data[seed]['icar'] = {}
        for di, delta in enumerate(deltas):
            data[seed]['icar'][di] = {}
            for gi, gamma in enumerate(gammas):
                data[seed]['icar'][di][gi] = {}
                for ei, epsilon in enumerate(epsilons):
                    data[seed]['icar'][di][gi][ei] = {}
                    for ci, config in enumerate(configs):
                        data_save_path = "dat/gap_experiment_{}_{}_{}_seed={}_icar_{}_{}_{}_{}.p".format(args.dataset, args.seed, u_str, seed, delta, epsilon, gamma, ci)
                        only_car, baseline_mode, gbc = config
                        try:
                            out = pickle.load(open(data_save_path, 'rb'))                            
                            print("Loading existing data for ICAR seed={}, delta={}, epsilon={}, gamma={} ...".format(520+seed, delta, epsilon, gamma), flush=True)
                        except FileNotFoundError:                    
                            settings = {
                                'epsilon': epsilon,
                                'delta': delta,
                                'gamma': gamma,
                                'max_k': int(np.ceil(np.log(0.5/gamma) / np.log(2))),
                                'zeta': 0.05,
                                'only_car': only_car,
                                'baseline_mode': baseline_mode,
                                'guessbestconf': gbc,
                                'seed': 520 + seed
                            }
                            icar = ImpatientCapsAndRuns(env=env, **settings)
                            ret = icar.run()
                            out = {'i_star': ret[0], 'total_time': env.total_runtime}
                            env.reset()
                            pickle.dump(out, open(data_save_path, 'wb'))
                        
                        data[seed]['icar'][di][gi][ei][ci] = out
                        print("ICAR: seed= {}, delta={}, epsilon={}, gamma={} config={}. i_star={}, total_time={}".format(seed, delta, epsilon, gamma, ci, data[seed]['icar'][di][gi][ei][ci]['i_star'], data[seed]['icar'][di][gi][ei][ci]['total_time'] / day_in_s), flush=True)


        data_save_path = "dat/gap_experiment_{}_{}_{}_seed={}_utilities.p".format(args.dataset, args.seed, u_str, seed)
        try: 
            data[seed]['utilities'] = pickle.load(open(data_save_path, 'rb'))
            print("Loading existing average utility data ...", flush=True)
        except FileNotFoundError:
            print("Calculating average utilities ...", flush=True)
            u_vect = np.vectorize(lambda t: u_fn(t, **u_params), otypes=[float])
            data[seed]['utilities'] = [np.mean(u_vect(env._runtimes[i, :])) for i in range(env.num_configs)]
            pickle.dump(data[seed]['utilities'], open(data_save_path, 'wb'))


        data_save_path = "dat/gap_experiment_{}_{}_{}_seed={}_mean_runtimes.p".format(args.dataset, args.seed, u_str, seed)
        try: 
            data[seed]['mean_runtimes'] = pickle.load(open(data_save_path, 'rb'))
            print("Loading existing mean_runtimes data ...", flush=True)
        except FileNotFoundError:
            print("Calculating mean runtimes ...", flush=True) 
            runtimes = {}
            for i in range(env.num_configs):
                runtimes[i] = np.sort(env._runtimes[i, :])
            data[seed]['mean_runtimes'] = {}
            for di, delta in enumerate(deltas):
                q = int((1 - delta) * env.num_instances)
                data[seed]['mean_runtimes'][di] = []
                for i in range(env.num_configs):
                    rts = [min(t, q) for t in runtimes[i]]
                    data[seed]['mean_runtimes'][di].append(np.mean(rts))
            pickle.dump(data[seed]['mean_runtimes'], open(data_save_path, 'wb'))
   

        data_save_path = "dat/gap_experiment_{}_{}_{}_seed={}_mean_runtimes_uncapped.p".format(args.dataset, args.seed, u_str, seed)
        try: 
            data[seed]['mean_runtimes_uncapped'] = pickle.load(open(data_save_path, 'rb'))
            print("Loading existing mean_runtimes_uncapped data ...", flush=True)
        except FileNotFoundError:
            print("Calculating mean runtimes uncapped ...", flush=True) 
            data[seed]['mean_runtimes_uncapped'] = [np.mean(env._runtimes[i, :]) for i in range(env.num_configs)]
            pickle.dump(data[seed]['mean_runtimes_uncapped'], open(data_save_path, 'wb'))


    ###################################### Plotting ###################################### 
    print("Plotting utility ...")
    draw_plot_curve(data, 'oup', plot_colors['oup'], seeds, zorder=None)
    # draw_plot_curve(data, 'coup', 'dodgerblue', seeds, zorder=None)
    draw_plot_curve(data, 'coupnm', plot_colors['coup'], seeds, zorder=None)

    for Bi, B in enumerate(Bs):
        for ki, k in enumerate(ks_sh):
            xs = [data[seed]['sh'][B][k]['total_time'] / day_in_s for seed in seeds]
            ys = [1 - data[seed]['utilities'][data[seed]['sh'][B][k]['i_star']] / np.max(data[seed]['utilities']) for seed in seeds]
            plot_rectangle(xs, ys, plot_colors['sh'], label="SuccessiveHalving" if (Bi + ki) == 0 else None, zorder=3)

    for Ri, (eta, R) in enumerate(etas_and_Rs):
        for ki, k in enumerate(ks_hb):
            xs = [data[seed]['hb'][R][k]['total_time'] / day_in_s for seed in seeds]
            ys = [1 - data[seed]['utilities'][data[seed]['hb'][R][k]['i_star']] / np.max(data[seed]['utilities']) for seed in seeds]
            plot_rectangle(xs, ys, plot_colors['hb'], label="Hyperband" if (Ri + ki) == 0 else None, zorder=3)

    for ki, k in enumerate(acb_ks):
        for ai, alpha in enumerate(acb_alphas):
            if args.dataset == 'minisat':
                xs_u = [data[seed]['acbu9'][k][ai][2]['cpu_rt'] / day_in_s for seed in seeds]
                ys_u = [1 - data[seed]['utilities'][data[seed]['acbu9'][k][ai][0][0]] / np.max(data[seed]['utilities']) for seed in seeds]
            else:
                xs_u = [data[seed]['acbu10'][k][ai][2]['cpu_rt'] / day_in_s for seed in seeds]
                ys_u = [1 - data[seed]['utilities'][data[seed]['acbu10'][k][ai][0][0]] / np.max(data[seed]['utilities']) for seed in seeds]
            plot_rectangle(xs_u, ys_u, plot_colors['acbu'], label="AC-Band, with $u$" if (ki + ai) == 0 else None, zorder=3)
    
    if args.dataset == 'minisat':
        plt.legend(loc='upper left')
    plt.xticks(fontsize=fs['ticks'])
    # plt.xlabel("CPU time (days)", fontsize=fs['axis'])
    # plt.ylabel("Percent gap to optimal", fontsize=fs['axis'])
    plt.title("{}".format(args.dataset), fontsize=fs['title'])

    
    plt.ylim(-0.01, 1.01)

    if args.dataset == "minisat":
        plt.xlim(.03, plt.xlim()[-1])
        plt.ylabel("Percent gap to optimal", fontsize=fs['axis'])
    elif args.dataset == "cplex_rcw":
        plt.xlim(1, plt.xlim()[-1])
    elif args.dataset == "cplex_region":
        plt.xlim(1, plt.xlim()[-1])

    plt.xscale('log')
    plt.yticks([0, 1/4, 1/2, 3/4, 1], [0, 25, 50, 75, 100], fontsize=fs['ticks'])
    plt.savefig("img/gap_{}_{}_{}.pdf".format(u_str, args.dataset, args.seed), bbox_inches='tight')

    linthresh = .02
    plt.ylim(0, 1)
    plt.yscale('symlog', linthresh=linthresh)    
    plt.yticks([0, 1/100, 1/10, 1], [0, 1, 10, 100], fontsize=fs['ticks'])
    yaxis = plt.gca().yaxis
    yaxis.set_minor_locator(MinorSymLogLocator(linthresh, .01))

    plt.savefig("img/log/gap_{}_{}_{}_log.pdf".format(u_str, args.dataset, args.seed), bbox_inches='tight')
    plt.clf()

    print("Plotting runtime ...")
    for di, delta in enumerate(deltas):
        draw_plot_curve(data, 'oup', plot_colors['oup'], seeds, is_runtime=True, delta_i=di,  zorder=2)
        draw_plot_curve(data, 'coupnm', plot_colors['coup'], seeds, is_runtime=True, delta_i=di, zorder=3)
        draw_plot_curve(data, 'spc', plot_colors['spc'], seeds, is_runtime=True, delta_i=di, zorder=1)

        for ki, k in enumerate(acb_ks):
            for ai, alpha in enumerate(acb_alphas):
                if args.dataset == 'minisat':
                    xs_rt = [data[seed]['acb9'][k][ai][2]['cpu_rt'] / day_in_s for seed in seeds]
                    ys_rt = [1 - np.min(data[seed]['mean_runtimes_uncapped']) / data[seed]['mean_runtimes_uncapped'][data[seed]['acb9'][k][ai][0][0]] for seed in seeds]
                else:
                    xs_rt = [data[seed]['acb10'][k][ai][2]['cpu_rt'] / day_in_s for seed in seeds]
                    ys_rt = [1 - np.min(data[seed]['mean_runtimes_uncapped']) / data[seed]['mean_runtimes_uncapped'][data[seed]['acb10'][k][ai][0][0]] for seed in seeds]
                plot_rectangle(xs_rt, ys_rt, plot_colors['acb'], label="AC-Band, default" if (ki + ai) == 0 else None, zorder=3)

        for gi, gamma in enumerate(gammas):
            for ei, epsilon in enumerate(epsilons):
                for ci, config in enumerate(configs):
                    xs = [data[seed]['icar'][di][gi][ei][ci]['total_time'] / day_in_s for seed in seeds]
                    ys = [1 - np.min(data[seed]['mean_runtimes'][di]) / data[seed]['mean_runtimes'][di][data[seed]['icar'][di][gi][ei][ci]['i_star']] for seed in seeds]
                    plot_rectangle(xs, ys, plot_colors['icar'], label="ICAR" if (gi + ei + ci) == 0 else None, zorder=3)

        if args.dataset == 'minisat':
            plt.legend(loc='upper right')
        plt.xticks(fontsize=fs['ticks'])
        plt.xlabel("CPU time (days)", fontsize=fs['axis'])
        # plt.ylabel("Percent gap to optimal", fontsize=fs['axis'])
        # plt.title("{}".format(args.dataset), fontsize=fs['title'])

        plt.ylim(-0.01, 1.01)        

        if args.dataset == "minisat":
            plt.xlim(.03, plt.xlim()[-1])
            plt.ylabel("Percent gap to optimal", fontsize=fs['axis'])
        elif args.dataset == "cplex_rcw":
            plt.xlim(1, plt.xlim()[-1])
        elif args.dataset == "cplex_region":
            plt.xlim(1, plt.xlim()[-1])

        plt.xscale('log')
        
        plt.yticks([0, 1/4, 1/2, 3/4, 1], [0, 25, 50, 75, 100], fontsize=fs['ticks'])
        plt.savefig("img/gap_runtime_delta={}_{}_{}_{}.pdf".format(delta, u_str, args.dataset, args.seed), bbox_inches='tight')

        linthresh = .02
        plt.ylim(0, 1)
        plt.yscale('symlog', linthresh=linthresh)
        plt.yticks([0, 1/100, 1/10, 1], [0, 1, 10, 100], fontsize=fs['ticks'])
        yaxis = plt.gca().yaxis
        yaxis.set_minor_locator(MinorSymLogLocator(linthresh, .01))

        plt.savefig("img/log/gap_runtime_delta={}_{}_{}_{}_log.pdf".format(delta, u_str, args.dataset, args.seed), bbox_inches='tight')

        plt.clf()






