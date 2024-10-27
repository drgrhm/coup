import math
import numpy as np

from utils import day_in_s, choose_max


def sh(env, u, captime, B):
    n = env.num_configs
    logn = math.ceil(math.log2(n))
    S = dict((i, 0) for i in range(n))
    Rk = 0
    for k in range(logn):
        rk = math.floor(B / len(S) / logn)
        Rk += rk

        for i in S:
            runtimes = [env.run(i, j, captime) for j in range(Rk)]
            utilities = [u(t) for t in runtimes]
            S[i] = sum(utilities) / Rk

        S = sorted(S.items(), key=lambda item: item[1], reverse=True)
        S = S[:math.ceil(len(S) / 2)]
        S = dict(S)

        i_star = max(S, key=S.get)
        print(f"round k={k}. len(S)={len(S)}, i_star={i_star}, U_i_star={S[i_star]}")
        
    return {'i_star': i_star, 'total_time': env.total_time}

# def oup_alpha(m, k, n, delta):
#     if m == 0:
#         return 1
#     return min(math.sqrt(math.log(4 * 2.705808 * n * m**2 * (math.log2(k) + 1)**2 / delta) / 2 / m), 1)


# def oup_message(r, i, i_star, i_prime, epsilon_star, UCB, LCB, m, k, env):
#     return f"oup: iteration {r}, i={i:5}, i_star={i_star:5}, epsilon_star={epsilon_star:.4f}, ucb=[{np.min(UCB):.4f}, {UCB[i_prime]:.4f}], lcb=[{np.min(LCB):.4f}, {LCB[i_star]:.4f}], m=[{np.min(m)}, {np.max(m)}], k=[{np.min(k)}, {np.max(k)}], total_time={env.total_time / day_in_s:.8f}"


# def update_output(out, **kwargs):
#     out['i_stars'].append(kwargs['i_star'])
#     out['epsilon_stars'].append(kwargs['epsilon_star'])
#     out['total_time'].append(kwargs['total_time'])
#     out['total_times'].append(kwargs['total_times'])


# def oup(env, u, delta, k0=1, epsilon_min=0, n=None, m_max=float('inf'), max_time=float("inf"), doubling_condition="new", improved_tie_breaking=False, save_mod=500, print_mod=10000):
#     """ Optimistic Utilitarian Procrastination """

#     if n is None:
#         n = env.num_configs

#     print("oup: running with {} doubling condition on {} configs ... ".format(doubling_condition, n))

#     I = dict([(i, None) for i in range(n)])
#     F_hat = np.zeros(n)
#     U_hat = np.zeros(n)
#     UCB = np.ones(n)
#     LCB = np.zeros(n)
#     m = np.zeros(n, dtype=np.int64)
#     k = np.ones(n) * k0
#     alpha = np.ones(n)
#     epsilon_star = 1
#     i_star_last = -1
#     r = 0
#     out = {'i_stars': [], 'epsilon_stars': [], 'total_time': [], 'total_times': []}

#     while epsilon_star > epsilon_min:
        
#         if env.total_time >= max_time:
#             update_output(out, i_star=i_star, epsilon_star=epsilon_star, total_time=env.total_time / day_in_s, total_times=env.total_times)
#             print(oup_message(r, i, i_star, i_prime, epsilon_star, UCB, LCB, m, k, env) + " TOTAL TIME REACHED")
#             return out

#         i = np.argmax(UCB)
#         m[i] += 1 
#         if m[i] >= m_max:
#             if epsilon_min == 0: # not targeting a specific epsilon
#                 print("\nWARNING oup ran out of instances at m={}. returning current results\n".format(m[i]))
#                 update_output(out, i_star=i_star, epsilon_star=epsilon_star, total_time=env.total_time / day_in_s, total_times=env.total_times)
#                 print(oup_message(r, i, i_star, i_prime, epsilon_star, UCB, LCB, m, k, env) + " MAX INSTANCES REACHED")
#                 return out

#             else: # targeting specific epsilon
#                 print("\nERROR: oup ran out of instances at m={} before reaching epsilon={}, current epsilon={}\n".format(m[i], epsilon_min, epsilon_star))
#                 raise IndexError

#         if doubling_condition == "old":
#             dubcond = 2 * alpha[i] <= u(k[i]) * (1 - F_hat[i])
#         elif doubling_condition == "new":
#             dubcond = 2 * (1 - u(k[i])) * alpha[i] <= u(k[i]) * (1 - F_hat[i] + alpha[i])

#         if dubcond: # if its time to double k[i]
#             k[i] = 2 * k[i]
#             runtimes = [env.run(i, j, k[i]) for j in range(m[i])]
#             F_hat[i] = sum([1 if t < k[i] else 0 for t in runtimes]) / m[i]
#             U_hat[i] = sum(u(t) for t in runtimes) / m[i]
#         else: # otherwise, just run the next instance
#             runtime = env.run(i, m[i] - 1, k[i])
#             F_hat[i] = ((m[i] - 1) * F_hat[i] + (1 if runtime < k[i] else 0)) / m[i]
#             U_hat[i] = ((m[i] - 1) * U_hat[i] + u(runtime)) / m[i]
        
#         alpha[i] = oup_alpha(m[i], k[i], n, delta)
#         UCB[i] = min(U_hat[i] + (1 - u(k[i])) * alpha[i], UCB[i])
#         LCB[i] = max(U_hat[i] - alpha[i] - u(k[i]) * (1 - F_hat[i]), LCB[i])

#         if improved_tie_breaking:
#             i_star = choose_max(LCB, U_hat)
#         else:
#             i_star = np.argmax(LCB)

#         # a faster check for suboptimals:
#         if i_star == i_star_last and i_star != i: # LCB[i_star] has not changed since last round 
#             if UCB[i] < LCB[i_star]: # only i needs to be checked
#                 I.pop(i)
#                 UCB[i] = -float("inf")
#                 LCB[i] = -float("inf")
#         else: # LCB[i_star] has changed since last round
#             for _i in list(I): # check all _i
#                 if UCB[_i] < LCB[i_star]:
#                     I.pop(_i)
#                     UCB[_i] = -float("inf")
#                     LCB[_i] = -float("inf")
#             i_star_last = i_star
        
#         i_prime = np.argmax(UCB)
#         epsilon_star = UCB[i_prime] - LCB[i_star]

#         if r % save_mod == 0:
#             update_output(out, i_star=i_star, epsilon_star=epsilon_star, total_time=env.total_time / day_in_s, total_times=env.total_times)
#         if r % print_mod == 0:            
#             print(oup_message(r, i, i_star, i_prime, epsilon_star, UCB, LCB, m, k, env))
#         r += 1

#     update_output(out, i_star=i_star, epsilon_star=epsilon_star, total_time=env.total_time / day_in_s, total_times=env.total_times)
#     print(oup_message(r, i, i_star, i_prime, epsilon_star, UCB, LCB, m, k, env) + " FINISHED")
#     return out
