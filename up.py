import math
import numpy as np   

from utils import day_in_s


def up(env, u, delta, k0=1, epsilon_min=0, n=None, save_mod=1, max_time=float("inf"), doubling_condition="new"):
    """ Utilitarian Procrastination """

    if n is None:
        n = env.num_configs

    num_instances = env.num_instances
    
    I = dict([(i, None) for i in range(n)])
    F_hat = dict([(i, 0) for i in range(n)])
    U_hat = np.zeros(n)
    UCB = np.ones(n)
    LCB = np.zeros(n)
    alpha = dict([(i, 1) for i in range(n)])
    k = dict([(i, k0) for i in range(n)])
    
    epsilon_star = 1
    i_stars = []
    epsilon_stars = []
    num_configs_remaining = []
    total_time = []
    total_times = []

    m = 0

    i_star = 0

    print("up running with {} doubling condition. delta={}, epsilon_min={}, m0={}".format(doubling_condition, delta, epsilon_min, m))

    while len(I) > 1 and epsilon_star > epsilon_min:
        m = m + 1
        
        for i in I.keys():
            alpha[i] = math.sqrt(math.log(4 * 2.705808 * n * m**2 * (math.log2(k[i]) + 1)**2 / delta) / 2 / m)

            if env.total_time >= max_time:
                i_stars.append(i_star)
                epsilon_stars.append(epsilon_star)        
                num_configs_remaining.append(len(I))
                total_time.append(env.total_time / day_in_s)
                total_times.append(env.total_time)
                print("up ran out time. Returning.")
                return {"i_stars": i_stars, 
                        "epsilon_stars": epsilon_stars, 
                        "num_configs_remaining": num_configs_remaining, 
                        "total_time": total_time, 
                        "total_times": total_times}

            if m >= num_instances:
                if epsilon_min == 0: # not targeting a specific epsilon
                    print("\nWARNING up ran out of instances at m={}. returning current results\n".format(m))                    
                    return {"i_stars": i_stars, 
                            "epsilon_stars": epsilon_stars, 
                            "num_configs_remaining": num_configs_remaining, 
                            "total_time": total_time, 
                            "total_times": total_times}
                else: # targeting specific epsilon
                    print("\nERROR: up ran out of instances at m={} before reaching epsilon={}, current epsilon={}\n".format(m, epsilon_min, epsilon_star))
                    raise IndexError

            if doubling_condition == "new":
                dubcond = 2 * (1 - u(k[i])) * alpha[i] <= u(k[i]) * (1 - F_hat[i] + alpha[i])
            elif doubling_condition == "old":
                dubcond = 2 * alpha[i] <= u(k[i]) * (1 - F_hat[i])

            if dubcond: # double k, recompute runtimes
                k[i] = 2 * k[i]
                runtimes = [env.run(i, j, k[i]) for j in range(m)]
                F_hat[i] = sum([1 if t < k[i] else 0 for t in runtimes]) / m
                U_hat[i] = sum(u(t) for t in runtimes) / m
            else: # do only next run and constant time updates:
                runtime = env.run(i, m - 1, k[i])
                F_hat[i] = ((m - 1) * F_hat[i] + (1 if runtime < k[i] else 0)) / m
                U_hat[i] = ((m - 1) * U_hat[i] + u(runtime)) / m

            UCB[i] = min(U_hat[i] + (1 - u(k[i])) * alpha[i], UCB[i])
            LCB[i] = max(U_hat[i] - alpha[i] - u(k[i]) * (1 - F_hat[i]), LCB[i])

        i_star = np.argmax(LCB)

        for _i in list(I):
            if UCB[_i] < LCB[i_star]:
                I.pop(_i)
                UCB[_i] = -float("inf")
                LCB[_i] = -float("inf")

        i_prime = np.argmax(UCB)
        epsilon_star = UCB[i_prime] - LCB[i_star]
        
        if (m - 1) % save_mod == 0:  
            i_stars.append(i_star)
            epsilon_stars.append(epsilon_star)        
            num_configs_remaining.append(len(I))
            total_time.append(env.total_time / day_in_s)
            total_times.append(env.total_times)

        if (m < 50) or (m < 1000 and m % 10 == 0) or (m % 1000 == 0):
            print("up: done m={} runs of n={} algorithms. k_min={}, k_max={}, epsilon_star={:.4f}, total_time_so_far={:.4f}, i_star={}, max_LCB={:.4f}, max_UCB={:.4f}".format(m, len(I), k[min(k, key=k.get)], k[max(k, key=k.get)], epsilon_star, env.total_time / day_in_s, i_star, LCB[i_star], UCB[i_prime]))
    
    i_stars.append(i_star)
    epsilon_stars.append(epsilon_star)        
    num_configs_remaining.append(len(I))
    total_time.append(env.total_time / day_in_s)
    total_times.append(env.total_times)
    
    return {"i_stars": i_stars, 
            "epsilon_stars": epsilon_stars, 
            "num_configs_remaining": num_configs_remaining, 
            "total_time": total_time, 
            "total_times": total_times}


