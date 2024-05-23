import math
import pickle
import numpy as np

from utils import *


class Environment():
    
    def __init__(self, data_file, max_timeout):
        if data_file is not None:
            self._runtimes = np.load(data_file)
        self._num_configs = self._runtimes.shape[0]
        self._num_instances = self._runtimes.shape[1]
        self._max_timeout = max_timeout

        ## Warning: shuffling can affect cumulative values (means) due to floating point precision.
        ## Should really shuffle ids instead of runtimes for numerical stability...
        self._runtimes = self._runtimes[:, np.random.permutation(self._num_instances)] # Randomlly shuffle instances
        self._runtimes = self._runtimes[np.random.permutation(self._num_configs), :] # Randomlly shuffle configurations         
        self.reset()

    def reset(self):
        self._completed = dict([(i, {}) for i in range(self._num_configs)]) 
        self._total_time = np.zeros(self._num_configs)
        # self._captimes_used = {}

    def get_num_configs(self):
        return self._num_configs

    def get_num_instances(self):
        return self._num_instances

    def get_max_timeout(self):
        return self._max_timeout

    def get_time_spent_running(self, i):
        return self._total_time[i]

    def get_time_spent_running_all(self):
        return sum(self._total_time)

    def get_total_time_per_config(self):
        return np.copy(self._total_time)

    def get_true_runtimes(self, n, m):
        """ return the runtime of the first n configs on the first m instances """
        return self._runtimes[:n, :m]

    def run(self, i, j, k):
        """ run configuration i on instance j with captime k """
        assert k <= self._max_timeout, "ERROR: captime k={} is greater than max for dataset".format(k)
        t = self._runtimes[i, j]  # uncapped runtime 
        if j in self._completed[i]:  # already completed this instance, no need to run again or do runtime accounting
            return min(t, k)
        else:  # do the run 
            if t < k:  # completed the run
                self._completed[i][j] = True
                self._total_time[i] += t
                return t
            else:  # run capped
                self._total_time[i] += k
                return k


class LBEnvironment(Environment):

    def __init__(self, data_file, max_timeout):
        data = pickle.load(open(data_file, 'rb'))
        self._runtimes = np.array([[min(t, max_timeout) for t in data[i]] for i in sorted(data.keys())])
        Environment.__init__(self, None, max_timeout)


    

