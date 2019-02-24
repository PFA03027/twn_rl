# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 15:33:39 2019

@author: alpha
"""

import argparse
import sys
from enum import Enum
import multiprocessing as mp
import logging


import myenv
import gym  #倒立振子(cartpole)の実行環境
from gym import wrappers  #gymの画像保存
import numpy as np
import time
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I

import chainerrl
from chainerrl import links
#from chainerrl.q_function import StateQFunction
import chainerrl.q_functions as Qfunc
#from chainerrl.action_value import DiscreteActionValue

import success_buffer_replay


def nl(x):
    return F.dropout(F.leaky_relu(x))

class Qfunc_FC_TWN(Qfunc.SingleModelStateQFunctionWithDiscreteAction):
    """Fully-connected state-input Q-function with discrete actions.

    Args:
        n_dim_obs: number of dimensions of observation space
        n_dim_action: number of dimensions of action space
        n_hidden_channels: number of hidden channels
        n_hidden_layers: number of hidden layers
    """
    class Qfunc_FC_TWN_model1(chainer.Chain):
        """ 共有メモリのctype arrayを直接numpyのバッファにする方法：　numpy.ctypeslib.as_arrayを使って、下記のようにするらしい。
        
        https://stackoverflow.com/questions/5549190/is-shared-readonly-data-copied-to-different-processes-for-multiprocessing/5550156#5550156
        shared_array_base = multiprocessing.Array(ctypes.c_double, 10*10)
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_array = shared_array.reshape(10, 10)

        numpyのドキュメントから、reshapeは必須。
        chainer.linksクラスに適用できる気がしない。。。。
        
        """
    
        def calc_num_out_elements1D(self, num_element, in_channel, out_channel, filter_size, slide_size, pooling_size):
            num_of_conv_out_elements = (num_element - filter_size)//slide_size + 1
            num_of_out_elements = num_of_conv_out_elements//pooling_size
            if (num_of_conv_out_elements%pooling_size) != 0:
                num_of_out_elements += 1
            
            return num_of_out_elements
        
        def calc_num_out_elementsND(self, num_element, pad_size, in_channel, out_channel, filter_size, slide_size, pooling_size):
            '''
            num_element: list type [180, 10]
            pad_size: list type [1, 1]
            
            return: type is np array
            '''
            np_num_element = np.array(num_element)
            np_pad_size = np.array(pad_size)
            num_of_conv_out_elements = ((np_num_element + np_pad_size) - filter_size)//slide_size + 1
            num_of_out_elements = num_of_conv_out_elements//pooling_size
            num_of_out_elements[num_of_conv_out_elements%pooling_size != 0] += 1
            
            return num_of_out_elements
        
        def __init__(self, n_size_twn_status, num_ray, n_size_eb_status, n_actions, explor_rate=0.0):
    
            self.n_size_twn_status = n_size_twn_status
            self.num_ray = num_ray
            self.n_size_eb_status = n_size_eb_status
            self.num_history = 1
            
    
            self.n_clasfy_ray = 16
    
            self.in_channel_1st = 1
            out_channel_1st = 16
            filter_size_1st = 5
            slide_size_1st = 1
            self.pooling_size_1st = 2
    
            out_channel_2nd = 64
            filter_size_2nd = 3
            slide_size_2nd = 1
            self.pooling_size_2nd = 4
            
            self.num_of_out_elements_1st = self.calc_num_out_elements1D(self.num_ray, self.in_channel_1st, out_channel_1st, filter_size_1st, slide_size_1st, self.pooling_size_1st)
            self.num_of_out_elements_2nd = self.calc_num_out_elements1D(self.num_of_out_elements_1st, out_channel_1st, out_channel_2nd, filter_size_2nd, slide_size_2nd, self.pooling_size_2nd)
                
            #self.logger.info('1st out: {}  2nd out: {}  n_actions'.format(self.num_of_out_elements_1st, self.num_of_out_elements_2nd, n_actions))
            print('1st out: {}  2nd out: {}  n_actions: {}'.format(self.num_of_out_elements_1st, self.num_of_out_elements_2nd, n_actions))
        
            super().__init__()
    
            with self.init_scope():
                self.conv1 = L.ConvolutionND(1, self.in_channel_1st, out_channel_1st, filter_size_1st) # 1層目の畳み込み層（チャンネル数は16）
                self.conv2 = L.ConvolutionND(1, out_channel_1st, out_channel_2nd, filter_size_2nd) # 2層目の畳み込み層（チャンネル数は64）
                self.l3 = L.Linear(self.num_of_out_elements_2nd*out_channel_2nd, self.n_clasfy_ray) #クラス分類用
                aaa = n_size_twn_status+self.n_clasfy_ray+n_size_eb_status
                self.l4 = L.Linear(aaa, aaa) #クラス分類用
                self.ml5 = links.MLP(aaa, n_actions, (aaa*3, aaa*2, (aaa//2)*3, aaa, aaa//2), nonlinearity=F.leaky_relu)
    
            self.explor_rate = explor_rate
            
            self.debug_info = None
    
    
        def __call__(self, state):
            '''
            state: state vector
            Q値の範囲が報酬体系によって負の値をとる場合、F.reluは負の値をとれないので、学習に適さない。
            活性化関数は、負の値も取ることが可能なものを選択する必要がある。
            例えば、F.leaky_relu等。勾配消失問題を考えると、これが良い感じ。
            
            return:
                type: Variable of Chainer
                Q values of all actions
            '''
    
            state32 = state.astype(np.float32)
    
            nrow, ncol = state32.shape
            #print('nrow, ncol =', nrow, ncol)
            twn_status = chainer.Variable(state32[:,0:self.n_size_twn_status].astype(np.float32))
            x_ray = state32[:,self.n_size_twn_status:self.n_size_twn_status+self.num_ray]
            eb_status = chainer.Variable(state32[:,self.n_size_twn_status+self.num_ray:self.n_size_twn_status+self.num_ray+self.n_size_eb_status].astype(np.float32))
            x = x_ray.reshape(nrow, self.in_channel_1st, self.num_ray)
    
            h1 = F.max_pooling_nd(F.leaky_relu(self.conv1(x)), self.pooling_size_1st) # 最大値プーリングは2×2，活性化関数はReLU
            h2 = F.max_pooling_nd(F.leaky_relu(self.conv2(h1)),self.pooling_size_2nd) 
            h3 = self.l3(h2)
            h3_c = F.concat((twn_status, h3, eb_status), axis=1)
            h4 = F.leaky_relu(self.l4(h3_c))
            h7 = self.ml5(h4)
            
            self.debug_info = (h7, h3, h1, h2, h3_c, h4)
    
            return h7

    class Qfunc_FC_TWN_model2(chainer.Chain):
        """ 
        """
        
        
        def __init__(self, n_size_twn_status, num_ray, n_size_eb_status, n_actions, explor_rate=0.0):
    
            self.n_size_twn_status = n_size_twn_status
            self.num_ray = num_ray
            self.n_size_eb_status = n_size_eb_status
            self.num_history = 1
            
    
            self.n_clasfy_ray = 32
    
            super().__init__()
    
            with self.init_scope():
                self.ml1 = links.MLP(self.num_ray, self.n_clasfy_ray, ((self.num_ray//3)*2, self.num_ray//2, self.num_ray//3), nonlinearity=F.leaky_relu)
                aaa = n_size_twn_status+self.n_clasfy_ray+n_size_eb_status
                self.l4 = L.Linear(aaa, aaa) #クラス分類用
                self.ml5 = links.MLP(aaa, n_actions, (aaa, aaa, aaa, aaa, aaa), nonlinearity=nl)
    
            self.explor_rate = explor_rate
            
            self.debug_info = None
    
    
        def __call__(self, state):
            '''
            state: state vector
            Q値の範囲が報酬体系によって負の値をとる場合、F.reluは負の値をとれないので、学習に適さない。
            活性化関数は、負の値も取ることが可能なものを選択する必要がある。
            例えば、F.leaky_relu等。勾配消失問題を考えると、これが良い感じ。
            
            return:
                type: Variable of Chainer
                Q values of all actions
            '''
    
            state32 = state.astype(np.float32)
    
            nrow, ncol = state32.shape
            #print('nrow, ncol =', nrow, ncol)
            twn_status = chainer.Variable(state32[:,0:self.n_size_twn_status].astype(np.float32))
            x_ray = state32[:,self.n_size_twn_status:self.n_size_twn_status+self.num_ray]
            eb_status = chainer.Variable(state32[:,self.n_size_twn_status+self.num_ray:self.n_size_twn_status+self.num_ray+self.n_size_eb_status].astype(np.float32))
            x = x_ray.reshape(nrow, self.num_ray)
            
            h1 = self.ml1(x)
            h3_c = F.concat((twn_status, h1, eb_status), axis=1)
            h4 = F.selu(self.l4(h3_c))
            h7 = self.ml5(h4)
            
            self.debug_info = (h7, h1, h3_c, h4)
    
            return h7


    def __init__(self, n_size_twn_status, num_ray, n_size_eb_status, n_actions, explor_rate=0.0):
        super().__init__(
                model=Qfunc_FC_TWN.Qfunc_FC_TWN_model1(
                        n_size_twn_status, num_ray, n_size_eb_status, n_actions, explor_rate=explor_rate
                        )
                )


def func_agent_generation(args, env, load_flag=False):
    gamma = 0.99
    alpha = 0.5

    # q_func = QFunction(env.observation_space.low.size, env.action_space.n)
    q_func = Qfunc_FC_TWN(env.obs_size_list[0], env.obs_size_list[1], env.obs_size_list[2], env.action_space.n)
    optimizer = chainer.optimizers.Adam(eps=1e-2)
    optimizer.setup(q_func)
    if load_flag:
        explorer = chainerrl.explorers.ConstantEpsilonGreedy(epsilon=0.05, random_action_func=env.action_space.sample)
    else:
        explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(start_epsilon=0.5, end_epsilon=0.05, decay_steps=50000, random_action_func=env.action_space.sample)
    #replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
    #replay_buffer = chainerrl.replay_buffer.PrioritizedReplayBuffer(capacity=10 ** 6)
    #replay_buffer = chainerrl.replay_buffer.PrioritizedEpisodicReplayBuffer(capacity=10 ** 6)
    replay_buffer = success_buffer_replay.SuccessPrioReplayBuffer(capacity=10 ** 6)
    phi = lambda x: x.astype(np.float32, copy=False)
    agent = chainerrl.agents.DoubleDQN(
        q_func, optimizer, replay_buffer, gamma, explorer,
        average_q_decay=0.01, average_loss_decay=0.01,
        update_interval=10, target_update_interval=200, phi=phi,
        replay_start_size=1500, minibatch_size=500,
        #replay_start_size=5, minibatch_size=3, episodic_update=True, episodic_update_len=64
        )

    #if len(args.load) > 0:
    if load_flag:
        #agent.load(args.load)
        agent.load('agent_ddqn')
        #logger.debug('load: {}'.format(args.load) )
        print('load: {}'.format(args.load) )

    return agent

