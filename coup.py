import math
import numpy as np   

from utils import *


def _alpha(m, k, n, delta):
    if m == 0:
        return 1
    return math.sqrt(math.log(4 * 2.705808 * n * m**2 * (math.log2(k) + 1)**2 / delta) / 2 / m)


def oup_message(r, i, i_star, i_prime, epsilon_star, UCB, LCB, k, m, env):
    print_string = ("oup: iteration {}, "
                    "i={:6}, "
                    "i_star={}, "
                    "epsilon_star={:.4f}, "
                    "ucb_max={:.4f}, "
                    "lcb_max={:.4f}, "
                    "k_min={}, "
                    "k_max={}, "
                    "m_min={}, "
                    "m_max={}, "
                    "total_time={:.8f}")
    print_string = print_string.format(r, 
                                       i, 
                                       i_star, 
                                       epsilon_star, 
                                       UCB[i_prime], 
                                       LCB[i_star], 
                                       k[min(k, key=k.get)], 
                                       k[max(k, key=k.get)], 
                                       m[min(m, key=m.get)], 
                                       m[max(m, key=m.get)], 
                                       env.get_time_spent_running_all() / day_in_s)
    return print_string


def oup(env, u, delta, k0=1, epsilon_min=0, n=None, m_max=float('inf'), save_mod=1000, doubling_condition="new"):
    """ Optimistic Utilitarian Procrastination """

    if n is None:
        n = env.get_num_configs()

    print("oup: running with {} doubling condition on {} configs ... ".format(doubling_condition, n))

    I = dict([(i, None) for i in range(n)])
    F_hat = dict([(i, 0) for i in range(n)])
    U_hat = dict([(i, 0) for i in range(n)])
    UCB = np.ones(n)
    LCB = np.zeros(n)
    m = dict([(i, 0) for i in range(n)])
    k = dict([(i, k0) for i in range(n)])
    alpha = dict([(i, 1) for i in range(n)])
    
    out = {'i_stars': [],
           'epsilon_stars': [],
           'num_configs_remaining': [],
           'total_times': [],
           'total_times_by_config': [],
           'max_k': [],
           'min_k': [],
           }

    epsilon_star = 1
    i_star_last = -1
    r = 0
    while epsilon_star > epsilon_min:
        
        i = np.argmax(UCB)
        m[i] += 1 
        if m[i] >= m_max:
            if epsilon_min == 0: # not targeting a specific epsilon
                print("\nWARNING oup ran out of instances at m={}. returning current results\n".format(m[i]))
                out['i_stars'].append(i_star)
                out['epsilon_stars'].append(epsilon_star)
                out['total_times'].append(env.get_time_spent_running_all() / day_in_s)
                out['total_times_by_config'].append(env.get_total_time_per_config())
                out['max_k'].append(k[max(k, key=k.get)])
                out['min_k'].append(k[min(k, key=k.get)])
                print("oup: iteration {}, i={:6}, i_star={}, epsilon_star={:.4f}, max_ucb={:.4f}, max_lcb={:.4f}, k_min={}, k_max={}, total_time={:.4f}, num_configs_remaining={}. FINISHED".format(r, i, i_star, epsilon_star, UCB[i_prime], LCB[i_star], k[min(k, key=k.get)], k[max(k, key=k.get)], env.get_time_spent_running_all() / day_in_s, len(I)))
                return out

            else: # targeting specific epsilon
                print("\nERROR: oup ran out of instances at m={} before reaching epsilon={}, current epsilon={}\n".format(m[i], epsilon_min, epsilon_star))
                raise IndexError

        if doubling_condition == "old":
            dubcond = 2 * alpha[i] <= u(k[i]) * (1 - F_hat[i])
        elif doubling_condition == "new":
            dubcond = 2 * (1 - u(k[i])) * alpha[i] <= u(k[i]) * (1 - F_hat[i] + alpha[i])

        if dubcond: # if its time to double k[i]
            k[i] = 2 * k[i]
            runtimes = [env.run(i, j, k[i]) for j in range(m[i])] 
            F_hat[i] = sum([1 if t < k[i] else 0 for t in runtimes]) / m[i]
            U_hat[i] = sum(u(t) for t in runtimes) / m[i]
        else: # otherwise, just run the next instance
            runtime = env.run(i, m[i] - 1, k[i])
            F_hat[i] = ((m[i] - 1) * F_hat[i] + (1 if runtime < k[i] else 0)) / m[i]
            U_hat[i] = ((m[i] - 1) * U_hat[i] + u(runtime)) / m[i]
        
        alpha[i] = _alpha(m[i], k[i], n, delta)
        UCB[i] = min(U_hat[i] + (1 - u(k[i])) * alpha[i], UCB[i])
        LCB[i] = max(U_hat[i] - alpha[i] - u(k[i]) * (1 - F_hat[i]), LCB[i])

        i_star = np.argmax(LCB) # try to eliminate configurations against LCB[i_star]:

        # a faster check for suboptimals:
        if i_star == i_star_last and i_star != i: # LCB[i_star] has not changed since last round 
            if UCB[i] < LCB[i_star]: # only i needs to be checked
                I.pop(i)
                UCB[i] = -float("inf")
                LCB[i] = -float("inf")
        else: # LCB[i_star] has changed since last round
            for _i in list(I): # check all _i
                if UCB[_i] < LCB[i_star]:
                    I.pop(_i)
                    UCB[_i] = -float("inf")
                    LCB[_i] = -float("inf")
            i_star_last = i_star
        
        i_prime = np.argmax(UCB)
        epsilon_star = UCB[i_prime] - LCB[i_star]

        if r % save_mod == 0:
            out['i_stars'].append(i_star)
            out['epsilon_stars'].append(epsilon_star)
            out['total_times'].append(env.get_time_spent_running_all() / day_in_s)
            out['total_times_by_config'].append(env.get_total_time_per_config())
            out['max_k'].append(k[max(k, key=k.get)])
            out['min_k'].append(k[min(k, key=k.get)])
        if r % 5000 == 0:            
            print(oup_message(r, i, i_star, i_prime, epsilon_star, UCB, LCB, k, m, env))
        r += 1

    out['i_stars'].append(i_star)
    out['epsilon_stars'].append(epsilon_star)
    out['total_times'].append(env.get_time_spent_running_all() / day_in_s)
    out['total_times_by_config'].append(env.get_total_time_per_config())
    out['max_k'].append(k[max(k, key=k.get)])
    out['min_k'].append(k[min(k, key=k.get)])
    print(oup_message(r, i, i_star, i_prime, epsilon_star, UCB, LCB, k, m, env) + " FINISHED")
    return out



def alpha_p(p, n, m, k, delta):
    return math.sqrt(math.log(36 * p**2 * n * m**2 * (math.log2(k) + 1)**2 / delta) / 2 / m)


def coup_message(p, r, n_p, i_star, epsilon_star, max_ucb, min_ucb, m, env):

    print_string = ("coup: phase p={}. "
                    "r={}. "
                    "n_p={} configs sampled. "
                    "i_star={}, "
                    "epsilon_star={:.4f}, "
                    "max_ucb={:.4f}, "
                    "max_lcb={:.4f}, "
                    "min_m={}, "
                    "max_m={}, "
                    "total_time={:.4f}")

    print_string = print_string.format(p, 
                                       r, 
                                       n_p, 
                                       i_star, 
                                       epsilon_star, 
                                       max_ucb, 
                                       min_ucb,
                                       m[min(m, key=m.get)],
                                       m[max(m, key=m.get)],
                                       env.get_time_spent_running_all() / day_in_s)    
    return print_string


def coup(env, u, delta, epsilon_fn, gamma_fn, k0=1, max_phases=float('inf'), n_max=float('inf'), m_max=float('inf'), doubling_condition="new"):
    """ Continuous, Optimistic Utilitarian Procrastination """

    F_hat = {}
    U_hat = {}
    m = {}
    k = {}
    ns = [0] # number of configs per phase 

    out = {'phase': [],
           'i_stars': [],
           'epsilon_stars': [],
           'max_num_instances': []
           }

    p = 1 # phase counter 
    while p <= max_phases:

        n_p = math.ceil(math.log(math.pi**2 * p**2 / 3 / delta) / gamma_fn(p))
        if n_p >= n_max:
            print("\nWARNING: coup needs n_p={} >= n_max={} configurations for phase p={}. returning phase {} results.\n".format(n_p, n_max, p, p-1))
            return out

        UCB = np.ones(n_p)
        LCB = np.zeros(n_p)

        for i in range(ns[-1]): # updates for existing configs
            if m[i] >= 1: # if we've run i before
                UCB[i] = min(U_hat[i] + (1 - u(k[i])) * alpha_p(p, n_p, m[i], k[i], delta), UCB[i])
                LCB[i] = max(U_hat[i] - alpha_p(p, n_p, m[i], k[i], delta) - u(k[i]) * (1 - F_hat[i]), LCB[i])

        for i in range(ns[-1], n_p): # initializations for new configs
            F_hat[i] = 0
            U_hat[i] = 0
            m[i] = 0
            k[i] = k0
            
        ns.append(n_p)

        i_prime = np.argmax(UCB)
        i_star = np.argmax(LCB)
        epsilon_star = 1

        r = 0 # loop counter for this phase
        while UCB[i_prime] - LCB[i_star] >= epsilon_fn(p):
            
            i = np.argmax(UCB)
            m[i] += 1
            if m[i] >= m_max:
                print("\nWARNING: coup reached m_max={} samples at round r={} in phase p={}. returning.\n".format(m_max, r, p))
                return out

            alpha_i = alpha_p(p, n_p, m[i], k[i], delta)
            
            if doubling_condition == "old":
                dubcond = 2 * alpha_i <= u(k[i]) *(1 - F_hat[i])
            elif doubling_condition == "new":
                dubcond = 2 * (1 - u(k[i])) * alpha_i <= u(k[i]) * (1 - F_hat[i] + alpha_i)

            if dubcond:
                k[i] = 2 * k[i]
                runtimes = [env.run(i, j, k[i]) for j in range(m[i])] 
                F_hat[i] = sum([1 if t < k[i] else 0 for t in runtimes]) / m[i]
                U_hat[i] = sum(u(t) for t in runtimes) / m[i]
            else: # otherwise, just run the next instance
                runtime = env.run(i, m[i] - 1, k[i])
                F_hat[i] = ((m[i] - 1) * F_hat[i] + (1 if runtime < k[i] else 0)) / m[i]
                U_hat[i] = ((m[i] - 1) * U_hat[i] + u(runtime)) / m[i]

            alpha_i = alpha_p(p, n_p, m[i], k[i], delta)
            UCB[i] = min(U_hat[i] + (1 - u(k[i])) * alpha_i, UCB[i])
            LCB[i] = max(U_hat[i] - alpha_i - u(k[i]) * (1 - F_hat[i]), LCB[i])

            i_star = np.argmax(LCB)
            i_prime = np.argmax(UCB)
            epsilon_star = UCB[i_prime] - LCB[i_star]

            # if r % save_mod == 0:
            #     out['i_stars'].append(i_star)
            #     out['epsilon_stars'].append(epsilon_star)
            #     out['max_num_instances'].append(max(m, key=m.get))
            #     out['total_times'].append(env.get_time_spent_running_all() / day_in_s)
            #     out['total_times_by_config'].append(np.copy(env._total_time))

            if r % 10000 == 0:
                print(coup_message(p, r, n_p, i_star, epsilon_star, UCB[i_prime], LCB[i_star], m, env))
            r += 1
        print(coup_message(p, r, n_p, i_star, epsilon_star, UCB[i_prime], LCB[i_star], m, env) + ". PHASE {} COMPLETE.".format(p))

        out['phase'].append({
            'total_time': env.get_time_spent_running_all() / day_in_s,
            'num_configs': n_p,
            'epsilon': epsilon_star,
            'i_star': i_star,
            'max_num_instances': m[max(m, key=m.get)],
            'time_per_config': env.get_total_time_per_config(),
            })

        p += 1

    return out



