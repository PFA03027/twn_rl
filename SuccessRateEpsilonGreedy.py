# -*- coding: utf-8 -*-
"""
Created on 

@author: alpha
"""
from logging import getLogger

from chainerrl import explorer
from chainerrl import explorers

class SuccessRateEpsilonGreedy(explorers.LinearDecayEpsilonGreedy):
    """Epsilon-greedy with Success rate and linearyly decayed epsilon

    Args:
      start_epsilon: max value of epsilon
      end_epsilon: min value of epsilon
      decay_steps: how many steps it takes for epsilon to decay
      random_action_func: function with no argument that returns action
      logger: logger used
    """

    def __init__(self, start_epsilon, end_epsilon,
                 decay_steps, random_action_func, logger=getLogger(__name__)):
        super().__init__(start_epsilon, end_epsilon, decay_steps, random_action_func, logger)
        self.success_rate = 0.0
        self.sr_epsilon_gain = 0.4

    def compute_epsilon(self, t):
        ret_value = self.end_epsilon
        if t <= self.decay_steps:
            epsilon_diff = self.end_epsilon - self.start_epsilon
            ret_value = self.start_epsilon + epsilon_diff * (t / self.decay_steps)
        
        if ret_value < (self.success_rate * self.sr_epsilon_gain):
            ret_value = self.success_rate * self.sr_epsilon_gain
        
        return ret_value
    
    def set_success_rate(self, r):
        self.success_rate = r

    def __repr__(self):
        return 'SuccessRateEpsilonGreedy(epsilon={},success rate={})'.format(self.epsilon, self.success_rate)