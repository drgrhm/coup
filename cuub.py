import math
import numpy as np   
# from collections import deque
from operator import itemgetter

from utils import *


def cuub(env, u, delta, k0=1, epsilon_min=0, n=None):

    if n is None:
        n = env.get_num_configs()

    if hasattr(env, 'get_num_instances'):
        num_instances = env.get_num_instances()
    else:
        num_instances = float('inf')

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
           'total_times_by_config': []
           }
    
    epsilon_star = 1
    i_star_last = -1
    r = 0
    while epsilon_star > epsilon_min:
        
        i = np.argmax(UCB)
        m[i] += 1 

        if m[i] >= num_instances:
            if epsilon_min == 0: # not targeting a specific epsilon
                print("\nWARNING cuub ran out of instances at m={}. returning current results\n".format(m[i]))
                out['i_stars'].append(i_star)
                out['epsilon_stars'].append(epsilon_star)
                out['total_times'].append(env.get_time_spent_running_all() / day_in_s)
                out['total_times_by_config'].append(np.copy(env._total_time))
                print("ucb: iteration {}, i={:4}, i_star={}, UCB[i_prime]={:.4f}, LCB[i_star]={:.4f}, epsilon_star={:.4f}, k_min={}, k_max={}, total_time={:.4f}, num_configs_remaining={}. FINISHED".format(r, i, i_star, UCB[i_prime], LCB[i_star], epsilon_star, k[min(k, key=k.get)], k[max(k, key=k.get)], env.get_time_spent_running_all() / day_in_s, len(I)))
                return out

            else: # targeting specific epsilon
                print("\nERROR: cuub ran out of instances at m={} before reaching epsilon={}, current epsilon={}\n".format(m[i], epsilon_min, epsilon_star))
                raise IndexError

        if 2 * alpha[i] <= u(k[i]) * (1 - F_hat[i]): # if its time to double k[i], rerun all instances
            k[i] = 2 * k[i]
            runtimes = [env.run(i, j, k[i]) for j in range(m[i])] 
            F_hat[i] = sum([1 if t < k[i] else 0 for t in runtimes]) / m[i]
            U_hat[i] = sum(u(t) for t in runtimes) / m[i]
        else: # otherwise, just run the next instance
            runtime = env.run(i, m[i], k[i])
            F_hat[i] = ((m[i] - 1) * F_hat[i] + (1 if runtime < k[i] else 0)) / m[i]
            U_hat[i] = ((m[i] - 1) * U_hat[i] + u(runtime)) / m[i]
        alpha[i] = math.sqrt(math.log(4 * 2.705808 * n * m[i]**2 * (math.log2(k[i]) + 1)**2 / delta) / 2 / m[i])
        UCB[i] = min(U_hat[i] + (1 - u(k[i])) * alpha[i], UCB[i])
        LCB[i] = max(U_hat[i] - alpha[i] - u(k[i]) * (1 - F_hat[i]), LCB[i])

        i_star = np.argmax(LCB) # try to eliminate configurations against LCB[i_star]:         

        # speedup check for suboptimals:
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

        if r % 1000 == 0:
            out['i_stars'].append(i_star)
            out['epsilon_stars'].append(epsilon_star)
            out['total_times'].append(env.get_time_spent_running_all() / day_in_s)
            out['total_times_by_config'].append(np.copy(env._total_time))
        if r % 5000 == 0:
            print("cuub: iteration {}, i={:4}, i_star={}, UCB[i_prime]={:.4f}, LCB[i_star]={:.4f}, epsilon_star={:.4f}, k_min={}, k_max={}, total_time={:.4f}, num_configs_remaining={}".format(r, i, i_star, UCB[i_prime], LCB[i_star], epsilon_star, k[min(k, key=k.get)], k[max(k, key=k.get)], env.get_time_spent_running_all() / day_in_s, len(I)))
        r += 1

    out['i_stars'].append(i_star)
    out['epsilon_stars'].append(epsilon_star)
    out['total_times'].append(env.get_time_spent_running_all() / day_in_s)
    out['total_times_by_config'].append(np.copy(env._total_time))
    print("cuub: iteration {}, i={:4}, i_star={}, UCB[i_prime]={:.4f}, LCB[i_star]={:.4f}, epsilon_star={:.4f}, k_min={}, k_max={}, total_time={:.4f}, num_configs_remaining={}. FINISHED".format(r, i, i_star, UCB[i_prime], LCB[i_star], epsilon_star, k[min(k, key=k.get)], k[max(k, key=k.get)], env.get_time_spent_running_all() / day_in_s, len(I)))
    return out





