import math
import numpy as np

from utils import day_in_s, choose_max


def get_hyperparameter_configuration(n, n_total, env):
    return dict((i, 0) for i in range(n_total, n_total + n))
    

def run_then_return_val_loss(t, ri,  env, u, captime):
    
    # # ri is number of samples:
    # ri = math.ceil(ri)
    # runtimes = [env.run(t, j, captime) for j in range(ri)]
    # utilities = [u(t) for t in runtimes]
    # return sum(utilities) / ri # utility, not loss...

    # ri is total time::
    total_time = 0
    runtimes = []
    j = 0
    while total_time < ri:
        runtime = env.run(t, j, captime)
        runtimes.append(runtime)
        total_time += runtime
    utilities = [u(t) for t in runtimes]
    return sum(utilities) / ri # utility, not loss...


def top_k(T, ni_new):
    T = sorted(T.items(), key=lambda item: item[1], reverse=True)
    T = T[:ni_new]
    return dict(T)


def hyperband(env, u, captime, R, eta):
    i_stars = []
    n_total = 0
    smax = math.floor(math.log(R, eta))
    B = (smax + 1) * R
    for s in reversed(range(smax + 1)):
        n = math.ceil(B * eta**s / R / (s + 1))
        r = R / eta**s
        T = get_hyperparameter_configuration(n, n_total, env)
        n_total += n
        print(f"s={s}, n={n}, n_total={n_total}...")
        for i in range(s + 1):
            ni = math.floor(n / eta**i)
            ri = r * eta**i
            
            for t in T:
                T[t] = run_then_return_val_loss(t, ri, env, u, captime)

            ni_new = max(math.floor(ni / eta), 1)
            T = top_k(T, ni_new)

            # print(f"...ni={ni}, ri={ri}, |T|={len(T)}" )
            i_star = max(T, key=T.get)
            i_stars.append((i_star, T[i_star]))
            

    i_stars = dict(i_stars)
    i_star = max(i_stars, key=i_stars.get)
    print(f"i_star={i_star}, total_time={env.total_time / day_in_s}")
    return {'i_star': i_star, 'total_time': env.total_time}





