import math
import pickle
import numpy as np

from utils import *


class Environment():
    
    def __init__(self, data_file, max_timeout, seed=0):
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
        # self._total_time = [0 for _ in range(self._num_configs)]
        self._total_time = np.zeros(self._num_configs)

    def get_num_configs(self):
        return self._num_configs

    def get_num_instances(self):
        return self._num_instances

    def get_max_timeout(self):
        return self._max_timeout

    def get_time_spent_running(self, i):
        # return self._total_time[self._config_ids[i]]
        return self._total_time[i]

    def get_time_spent_running_all(self):
        return sum(self._total_time)

    def run(self, i, j, k):
        """ run configuration i on instance j with captime k """
        assert k <= self._max_timeout, "ERROR: captime k={} is greater than max for dataset".format(k)
        
        # t = self._runtimes[self._config_ids[i], j]  # uncapped runtime 
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


class SampledEnvironment(Environment):

    def __init__(self, data_file, max_timeout, num_configs):
        self._config_ids = np.random.permutation(self._runtimes.shape[0])[:num_configs]
        self._runtimes = np.load(data_file)[self._config_ids, :]
        Environment.__init__(self, None, max_timeout)


class SimulatedEnvironment():

    def __init__(self):
        self.reset_all()

    def reset_all(self): # reset time spent and discard all sampled configs/instances
        self._total_time = {}
        self._active_configs = {}
        self._active_instances = {}
        self._completed = {} # self._completed[i][j] is the time for config i to complete instance j
        self._pending = {} # self._pending[i][j] = t if i timed out the last time it was run on j
        self._sample_fn = {} 
        self._params = {}
        self._instance_multiplier = {}

    def reset(self): # only reset time spent, not configs/instances themselves
        self._total_time = {}
        for i in self._active_configs:
            self._total_time[i] = 0
            self._pending[i].update(self._completed[i])
            self._completed[i] = {}

    def get_time_spent_running(self, i):
        if i in self._active_configs:
            return self._total_time[i]
        else:
            return 0

    def get_time_spent_running_all(self):
        return sum(self._total_time.values())

    def run(self, i, j, k):
        """ run configuration i on instance j with captime k """
        
        if i not in self._active_configs: # new config, initialize values
            mu = np.random.lognormal(.5, 1)
            sig = np.random.lognormal(0, .25)
            self._params[i] = {'mu': mu, 'sig': sig}
            self._sample_fn[i] = lambda x: np.random.lognormal(x + mu, sig)
            self._completed[i] = {}
            self._pending[i] = {} # holds sampled runtime values until completed
            self._total_time[i] = 0
            self._active_configs[i] = True

        if j not in self._active_instances: # new instance 
            self._instance_multiplier[j] = np.random.lognormal(.5, 1)
            self._active_instances[j] = True

        if j in self._completed[i]: # already completed this instance, no need to run again or do runtime accounting
            return min(self._completed[i][j], k)
        else:
            if j in self._pending[i]: # ran i on j before and timed out
                t = self._pending[i][j]
            else: # new instance for config i
                t = self._sample_fn[i](self._instance_multiplier[j]) # sample new runtime value 
                self._pending[i][j] = t # save sampled runtime value

            if t < k:  # completed the run
                del self._pending[i][j] 
                self._completed[i][j] = t
                self._total_time[i] += t
                return t
            else:  # the run capped
                self._total_time[i] += k
                return k



