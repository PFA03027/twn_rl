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
        self.sr_epsilon_gain = 0.4
        self.exp_rate = self.sr_epsilon_gain

    def compute_epsilon(self, t):
        ret_value = super().compute_epsilon(t)
        
        if ret_value < self.exp_rate:
            ret_value = self.exp_rate
        
        return ret_value
    
    def set_success_rate(self, r):
        self.exp_rate = (1.0 - r) * self.sr_epsilon_gain

    def __repr__(self):
        return 'SuccessRateEpsilonGreedy(epsilon={},explorer rate={})'.format(self.epsilon, self.exp_rate)
