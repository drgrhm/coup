import math

from utils import day_in_s


def naive(env, u, epsilon, delta, k):
    """ Naive algorithm """
    assert u(k) < epsilon, "ERROR: captime must be large enough. k={}, u(k)={}, epsilon={}".format(k, u(k), epsilon)
    
    n = env.num_configs
    m = int(2 * math.log(2 * n / delta) / (epsilon - u(k))**2) + 1
    
    assert m < env.num_instances, "ERROR: cannot do m={} runs, only {} instances. u(k)={}, epsilon={}".format(m, env.num_instances, u(k), epsilon)
    
    print("naive: doing m={} runs of n={} algorithms at captime k={}. u(k)={}, epsilon={}".format(m, n, k, u(k), epsilon))

    U = {}
    for i in range(n):
        U[i] = sum(u(env.run(i, j, k)) for j in range(m)) / m
    i_star = max(U, key=U.get)
    print("naive: i_star={}, total_time={}".format(i_star, env.total_time / day_in_s))
    return {'i_star': i_star, 
            'm': m,
            'total_time': env.total_time / day_in_s}

