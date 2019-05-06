# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 15:36:16 2019

@author: alpha
"""


import argparse
import sys
from enum import Enum
import multiprocessing as mp
import logging
import os


import myenv
import gym  #倒立振子(cartpole)の実行環境
from gym import wrappers  #gymの画像保存
import numpy as np
import time
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
from chainer import serializers

import chainerrl
from chainerrl import agent
from chainerrl import links
#from chainerrl.q_function import StateQFunction
import chainerrl.q_functions as Qfunc
from chainerrl.action_value import DiscreteActionValue
from chainerrl.agents import dqn

import success_buffer_replay
import SuccessRateEpsilonGreedy
import twn_model_base


class Qfunc_FC_TWN2_Vision(Qfunc.StateQFunction, agent.AttributeSavingMixin):
    """AEによるレーダー情報分析層

    Args:
        num_ray: TWNセンサーでとらえた周囲の状況を表すレーダーの本数
        n_clasfy_ray: レーダー情報の分析結果の出力要素数
    """

    class Qfunc_FC_TWN_model_AECNN_for_Ray(chainer.Chain):
        """ 
        AutoEncoder of CNN for ray input
        レーダーセンサーの入力情報に対する畳み込み層のモデル
        畳み込み層は、特徴抽出ための層となるので、AutoEncoderの手法を利用して、学習を行う。
        よって、この層は、強化学習の対象とはならない。
        
        """
    
        def calc_num_out_elements1D(self, num_element, in_channel, out_channel, filter_size, slide_size, pooling_size):
            '''
            num_element: number of input elements per input channel
            in_channel: number of input channel
            out_channel: number of output channel
            filter_size: filter size of convolution
            slide_size: slide size of filter
            pooling_size: pooling size for output
            
            return of 1st: number of elements of convolution output
            return of 2nd: number of elements of pooling output
            '''
            num_of_conv_out_elements = (num_element - filter_size)//slide_size + 1
            num_of_pooling_out_elements = num_of_conv_out_elements//pooling_size
            if (num_of_conv_out_elements%pooling_size) != 0:
                num_of_pooling_out_elements += 1
            
            return num_of_conv_out_elements, num_of_pooling_out_elements
        
        def calc_num_out_elementsND(self, num_element, pad_size, in_channel, out_channel, filter_size, slide_size, pooling_size):
            '''
            num_element: list type [180, 10]
            pad_size: list type [1, 1]
            
            return: type is np array
            '''
            np_num_element = np.array(num_element)
            np_pad_size = np.array(pad_size)
            num_of_conv_out_elements = ((np_num_element + np_pad_size) - filter_size)//slide_size + 1
            num_of_pooling_out_elements = num_of_conv_out_elements//pooling_size
            num_of_pooling_out_elements[num_of_conv_out_elements%pooling_size != 0] += 1
            
            return num_of_conv_out_elements, num_of_pooling_out_elements
        
        def __init__(self, num_in_elements, num_in_channel=1, out_channel=16, filter_size=5, slide_size=1, pooling_size=2, name=None, dropout_rate=0.0):
            '''
            num_in_elements: number of input elements per input channel
            num_in_channel: number of input channel
            out_channel: number of output channel
            filter_size: filter size of convolution
            slide_size: slide size of filter
            pooling_size: pooling size for output
            Name: name of this layer
            dropout_rate: dropout ratio of output layer
            '''
    
            self.num_in_elements = num_in_elements
    
            self.in_channel = num_in_channel
            self.out_channel = out_channel
            self.filter_size = filter_size
            self.slide_size = slide_size
            self.pooling_size = pooling_size
            
            self.dropout_rate = dropout_rate
    
            self.num_of_conv_out_elements, self.num_of_pooling_out_elements = self.calc_num_out_elements1D(self.num_in_elements, self.in_channel, self.out_channel, self.filter_size, self.slide_size, self.pooling_size)

            print('Layer {}: in: {} drop rate: {:>4.1%} out: conv out {}  pooling out {}'.format(
                name,
                self.num_in_elements,
                self.dropout_rate,
                self.num_of_conv_out_elements,
                self.num_of_pooling_out_elements
                ))

            super().__init__()
            with self.init_scope():
                self.conv = L.ConvolutionND(1, self.in_channel, self.out_channel, ksize=self.filter_size, stride=self.slide_size)
                self.dcnv = L.DeconvolutionND(1, self.out_channel, self.in_channel, ksize=self.filter_size, stride=self.slide_size)
            
        def fwd(self, state):
            if self.dropout_rate == 0.0:
                h1 = F.relu(self.conv(state))
            else:
                h1 = F.dropout(F.relu(self.conv(state)), ratio=self.dropout_rate)
                
            return h1

        def fwd_loss(self, state):
            h1 = self.fwd(state)
            dcnv_out = self.dcnv(h1)
            loss = F.mean_squared_error(dcnv_out, state)

            return [h1, loss, dcnv_out, state]

        def __call__(self, state):
            '''
            強化学習用のQ関数ではないので、普通にlossを返す
            '''
            return self.fwd_loss(state)[1]

    class Qfunc_FC_TWN_model_AECNN_for_Clasfy(chainer.Chain):
        """ 
        レーダーセンサーの入力情報に対する畳み込み層出力後の全結合層
        特徴抽出ための層となるので、AutoEncoderの手法を利用して、学習を行う。
        よって、この層は、強化学習の対象とはならない。
        
        """
    
        def __init__(self, num_in_elements, num_in_channel, n_clasfy_ray, name=None, dropout_rate=0.0):
            '''
            num_in_elements: number of　output elements per output channel of CNN as input
            num_in_channel: number of　output channel of CNN as input
            n_clasfy_ray: number of clasify
            Name: name of this layer
            dropout_rate: dropout ratio of output layer. this is experimental purpose.
            '''

            self.num_in_elements = num_in_elements
            self.num_in_channel = num_in_channel
    
            self.n_clasfy_ray = n_clasfy_ray
    
            self.dropout_rate = dropout_rate
    
            super().__init__()
            with self.init_scope():
                self.l_in = L.Linear(self.num_in_elements*self.num_in_channel, self.n_clasfy_ray) #クラス分類用
                self.l_out = L.Linear(self.n_clasfy_ray, self.num_in_elements*self.num_in_channel) #クラス分類用
            
            print('Layer {}: in: element {}  channel {}, out: {}'.format(name, self.num_in_elements, self.num_in_channel, self.n_clasfy_ray))
            
        def fwd(self, state):
            if self.dropout_rate == 0.0:
                h1 = F.sigmoid(self.l_in(state))
            else:
                h1 = F.dropout(F.sigmoid(self.l_in(state)), ratio=self.dropout_rate)

            return h1

        def fwd_loss(self, state):
            h1 = self.fwd(state)
            dcnv_out = F.reshape(self.l_out(h1), state.shape)
            loss = F.mean_squared_error(dcnv_out, state)

            return [h1, loss, dcnv_out, state]

        def __call__(self, state):
            '''
            強化学習用のQ関数ではないので、普通にlossを返す
            '''
            return self.fwd_loss(state)[1]

    saved_attributes = ("model1", "model2", "model3")

    def __init__(self, num_ray, n_clasfy_ray):

        self.num_ray = num_ray
        self.n_clasfy_ray = n_clasfy_ray

        self.in_channel_1st = 1
        out_channel_1st = 16
        filter_size_1st = 5
        slide_size_1st = 1
        self.pooling_size_1st = 2

        out_channel_2nd = 64
        filter_size_2nd = 3
        slide_size_2nd = 1
        self.pooling_size_2nd = 4
        
        self.model1 = Qfunc_FC_TWN2_Vision.Qfunc_FC_TWN_model_AECNN_for_Ray(
                num_ray,
                num_in_channel=self.in_channel_1st,
                out_channel=out_channel_1st,
                filter_size=filter_size_1st,
                slide_size=slide_size_1st,
                pooling_size=self.pooling_size_1st,
                name="1st conv",
                dropout_rate=0.2)
        self.model2 = Qfunc_FC_TWN2_Vision.Qfunc_FC_TWN_model_AECNN_for_Ray(
                self.model1.num_of_pooling_out_elements,
                num_in_channel=out_channel_1st,
                out_channel=out_channel_2nd,
                filter_size=filter_size_2nd,
                slide_size=slide_size_2nd,
                pooling_size=self.pooling_size_2nd,
                name="2nd conv",
                dropout_rate=0.5)

        self.model3 = Qfunc_FC_TWN2_Vision.Qfunc_FC_TWN_model_AECNN_for_Clasfy(
                self.model2.num_of_pooling_out_elements,
                out_channel_2nd,
                self.n_clasfy_ray,
                name="Clasify")

        self.model_list = [self.model1, self.model2, self.model3]
        
        self.debug_info = None
        
    def fwd(self, x):
        h1 = F.max_pooling_nd(self.model1.fwd(x), self.pooling_size_1st)
        h2 = F.max_pooling_nd(self.model2.fwd(h1),self.pooling_size_2nd)
        h3 = self.model3.fwd(h2)
        
        return h3


    def __call__(self, x):
        '''
        強化学習用のQ関数ではないので、普通にlossを返す
        '''
        model1_out = self.model1.fwd_loss(x)
        h1 = F.max_pooling_nd(model1_out[0], self.pooling_size_1st)
        model2_out = self.model2.fwd_loss(h1)
        h2 = F.max_pooling_nd(model2_out[0], self.pooling_size_2nd)
        model3_out = self.model3.fwd_loss(h2)
        
        self.debug_info = (model1_out, model3_out, model3_out)

        return [model1_out[1], model2_out[1], model3_out[1]]
    
    def cleargrads(self):
        for m in self.model_list:
            m.cleargrads()
                
    def gen_setup_optimizer(self, opt_type):
        '''
        内部で抱えるそれぞれのモデルに対するそれぞれのOptimizerを生成する
        
        return:
            opt_type: chainer.optimizers.XXX
        '''
        ans = []
        assert issubclass(opt_type,chainer.Optimizer)
        for m in self.model_list:
            optimizer = opt_type()
            optimizer.setup(m)
            ans.append(optimizer)
        
        return ans

class Qfunc_FC_TWN_RL(Qfunc.SingleModelStateQFunctionWithDiscreteAction, agent.AttributeSavingMixin):
    """行動価値関数 = Q関数

    Args:
        n_size_twn_status: TWN自身の状態
        num_ray: TWNセンサーでとらえた周囲の状況
        n_size_eb_status: TWNセンサーでとらえた、EBの情報
        n_actions: 離散アクション空間
        explor_rate=0.0: 探索行動比率（現時点で未使用）
    """

    class Qfunc_FC_TWN_model_RLLayer(chainer.Chain, agent.AttributeSavingMixin):
        """ 
        強化学習の対象となる層
        """
        
        saved_attributes = ('l4','l5','action_chain_list')
    
        def __init__(self, n_in_elements, n_actions, explor_rate=0.0):
            '''
            Q値の範囲が報酬体系によって負の値をとる場合、F.reluは負の値をとれないので、学習に適さない。
            活性化関数は、負の値も取ることが可能なものを選択する必要がある。
            例えば、F.leaky_relu等。勾配消失問題を考えると、これが良い感じ。
            
            n_size_twn_status: TWN自身の状態
            num_ray_out: TWNセンサーをCNN層で処理した結果得られるデータ数のタプル (CNN層の出力要素数, CNN層の出力チャンネル数)
            n_size_eb_status: TWNセンサーでとらえた、EBの情報
            n_actions: 離散アクション空間
            explor_rate=0.0: 探索行動比率（現時点で未使用）
            '''

            super().__init__()
            with self.init_scope():
                self.l4 = links.MLP(n_in_elements, int(n_in_elements*1.2), (n_in_elements*2, int(n_in_elements*1.8), int(n_in_elements*1.5)), nonlinearity=F.leaky_relu)
                self.l5 = links.MLP(int(n_in_elements*1.2)+4, 4, (n_in_elements, int(n_in_elements*0.8), (n_in_elements*2)//3), nonlinearity=F.leaky_relu)
                local_action_links_list = []
                for i in range(n_actions):
                    action_links = links.MLP(4, 1, (n_in_elements//2,), nonlinearity=F.leaky_relu)
                    local_action_links_list.append(action_links)
                self.action_chain_list = chainer.ChainList(*local_action_links_list)

            self.explor_rate = explor_rate
            
            self.debug_info = None
    
    
        def __call__(self, x):
            '''
            return:
                type: Variable of Chainer
                Q values of all actions
            '''

            ss = list(x.shape)
            ss[1] = 1
            noise0 = np.random.normal( 0.0, 0.5, ss)
            noise1 = np.random.normal(-1.0, 0.5, ss)
            noise2 = np.random.normal( 1.0, 0.5, ss)
            noise3 = np.random.uniform(low=-1.0, high=1.0, size=ss)
            
            rdata = np.hstack([noise0, noise1, noise2, noise3]).astype(np.float32)

            h4 = self.l4(x)
            h4_c = F.concat([h4, chainer.Variable(rdata)], axis=1)
            h5 = self.l5(h4_c)
            action_Qvalues = []
            for act_mlp in self.action_chain_list:
                qout = act_mlp(h5)
                action_Qvalues.append(qout)
            
            h7 = F.concat(action_Qvalues, axis=1)
            
            self.debug_info = (h7)
    
            return h7


    saved_attributes = ('model',)

    def __init__(self, n_in_elements, n_actions, explor_rate=0.0):
        super().__init__(
                model = Qfunc_FC_TWN_RL.Qfunc_FC_TWN_model_RLLayer(
                        n_in_elements,
                        n_actions,
                        explor_rate=0.0
                        )
                )

class Qfunc_FC_TWN2_History(Qfunc.StateQFunction, agent.AttributeSavingMixin):
    """入力情報の履歴からの分析層

    Args:
        num_ray: TWNセンサーでとらえた周囲の状況を表すレーダーの本数
        n_clasfy_ray: レーダー情報の分析結果の出力要素数
    """
    class Qfunc_FC_TWN2_History_AE(chainer.Chain):
        """ 
        入力情報の履歴に分析用全結合層
        特徴抽出ための層となるので、AutoEncoderの手法を利用して、学習を行う。
        よって、この層は、強化学習の対象とはならない。
        
        """
    
        def __init__(self, num_in_elements, num_out_elements, name=None):
            '''
            num_in_elements: number of input elements as input
            num_out_elements: number of output elements as output
            Name: name of this layer
            '''

            self.num_in_elements = num_in_elements
            self.num_out_elements = num_out_elements
            self.layer_name = name
    
            super().__init__()
            with self.init_scope():
                self.l_in = L.Linear(self.num_in_elements, self.num_out_elements)   # 次元削減層
                self.l_out = L.Linear(self.num_out_elements, self.num_in_elements)  # 次元復元層
            
            print('Layer {}: in: {}, out: {}'.format(name, self.num_in_elements, self.num_out_elements))
            
        def fwd(self, state):
            h1 = F.leaky_relu(self.l_in(state))

            return h1

        def fwd_loss(self, state):
            h1 = self.fwd(state)
            dcnv_out = F.reshape(self.l_out(h1), state.shape)
            loss = F.mean_squared_error(dcnv_out, state)

            return [h1, loss, dcnv_out, state]

        def __call__(self, state):
            '''
            強化学習用のQ関数ではないので、普通にlossを返す
            '''
            return self.fwd_loss(state)[1]

    saved_attributes = ("model1", "model2", "model3", "model4", "model5", "model6")

    def __init__(self, num_input, history_num, num_output):
        self.num_input = num_input
        self.history_num = history_num
        self.num_output = num_output

        self.output_1st = int((self.num_input * self.history_num - self.num_output) * 0.7 + self.num_output)
        self.output_2nd = int((self.num_input * self.history_num - self.num_output) * 0.3 + self.num_output)
        self.output_3rd = int((self.num_input * self.history_num - self.num_output) * 0.1 + self.num_output)
        self.output_4th = int((self.num_input * self.history_num - self.num_output) * 0.07 + self.num_output)
        self.output_5th = int((self.num_input * self.history_num - self.num_output) * 0.03 + self.num_output)

        self.model1 = Qfunc_FC_TWN2_History.Qfunc_FC_TWN2_History_AE(
            self.num_input * self.history_num,
            self.output_1st,
            name="1st history analysis")
        self.model2 = Qfunc_FC_TWN2_History.Qfunc_FC_TWN2_History_AE(
            self.output_1st,
            self.output_2nd,
            name="2nd history analysis")
        self.model3 = Qfunc_FC_TWN2_History.Qfunc_FC_TWN2_History_AE(
            self.output_2nd,
            self.output_3rd,
            name="3rd history analysis")
        self.model4 = Qfunc_FC_TWN2_History.Qfunc_FC_TWN2_History_AE(
            self.output_3rd,
            self.output_4th,
            name="4th history analysis")
        self.model5 = Qfunc_FC_TWN2_History.Qfunc_FC_TWN2_History_AE(
            self.output_4th,
            self.output_5th,
            name="5th history analysis")
        self.model6 = Qfunc_FC_TWN2_History.Qfunc_FC_TWN2_History_AE(
            self.output_5th,
            self.num_output,
            name="6th history analysis")

        self.model_list = [self.model1, self.model2, self.model3, self.model4, self.model5, self.model6]
        
        self.debug_info = None

        
    def fwd(self, x):
        h1 = self.model1.fwd(x)
        h2 = self.model2.fwd(h1)
        h3 = self.model3.fwd(h2)
        h4 = self.model4.fwd(h3)
        h5 = self.model5.fwd(h4)
        h6 = self.model6.fwd(h5)
        
        return h6


    def __call__(self, x):
        '''
        強化学習用のQ関数ではないので、普通にlossを返す
        '''
        model1_out = self.model1.fwd_loss(x)
        model2_out = self.model2.fwd_loss(model1_out[0])
        model3_out = self.model3.fwd_loss(model2_out[0])
        model4_out = self.model4.fwd_loss(model3_out[0])
        model5_out = self.model5.fwd_loss(model4_out[0])
        model6_out = self.model6.fwd_loss(model5_out[0])
        
        self.debug_info = (model1_out, model2_out, model3_out, model4_out, model5_out, model6_out)

        return [model1_out[1], model2_out[1], model3_out[1], model4_out[1], model5_out[1], model6_out[1]]
    
    def cleargrads(self):
        for m in self.model_list:
            m.cleargrads()
                
    def gen_setup_optimizer(self, opt_type):
        '''
        内部で抱えるそれぞれのモデルに対するそれぞれのOptimizerを生成する
        
        return:
            opt_type: chainer.optimizers.XXX
        '''
        ans = []
        assert issubclass(opt_type,chainer.Optimizer)
        for m in self.model_list:
            optimizer = opt_type()
            optimizer.setup(m)
            ans.append(optimizer)
        
        return ans


class MMAgent_DDQN(agent.Agent, agent.AttributeSavingMixin, twn_model_base.TWNAgentMixin):
    """エージェント本体

    Args:
        num_ray: TWNセンサーでとらえた周囲の状況を表すレーダーの本数
        n_clasfy_ray: レーダー情報の分析結果の出力要素数
    """
    saved_attributes = ['cnn_ae', 'q_func', 'q_func_opt', 'hist_ana_ae']

    def __init__(self, args, env, load_flag=False, explor_rate=None):
        super().__init__()

        self.n_size_twn_status = env.obs_size_list[0]
        self.num_ray           = env.obs_size_list[1]
        self.n_size_eb_status  = env.obs_size_list[2]

        self.update_interval = 10
        self.target_update_interval = 200
        self.replay_start_size = 1000
        self.minibatch_size = 512

        self.history_num = 30
        self.history_update_interval = 15
        self.history_append_count = 0
        self.history_data = []

        self.success_rate = 1.0

        gamma = 0.99
        alpha = 0.5
        
        n_clasfy_ray = 32

#        self.q_func = Qfunc_FC_TWN2_Vision(env.obs_size_list[0], env.obs_size_list[1], env.obs_size_list[2], env.action_space.n)
        self.cnn_ae = Qfunc_FC_TWN2_Vision(self.num_ray, n_clasfy_ray)
        self.cnn_ae_opts = self.cnn_ae.gen_setup_optimizer(chainer.optimizers.Adam)
        self.replay_buffer_cnn_ae = success_buffer_replay.SuccessPrioReplayBuffer(capacity=10 ** 6)
        self.cnn_ae_last_losses = None

        self.hist_ana_ae = Qfunc_FC_TWN2_History(self.n_size_twn_status + n_clasfy_ray + self.n_size_eb_status, self.history_num, self.n_size_twn_status + n_clasfy_ray + self.n_size_eb_status)
        self.hist_ana_ae_opts = self.hist_ana_ae.gen_setup_optimizer(chainer.optimizers.Adam)
        self.replay_buffer_hist_ana_ae = chainerrl.replay_buffer.ReplayBuffer(capacity=5000)
        self.hist_ana_ae_last_losses = None
        self.hist_ana_ae_last_out = None

        self.q_func = Qfunc_FC_TWN_RL((self.n_size_twn_status + n_clasfy_ray + self.n_size_eb_status)*2, env.action_space.n)
        self.q_func_opt = chainer.optimizers.Adam(eps=1e-2)
        self.q_func_opt.setup(self.q_func)
        self.explorer = None
        if load_flag:
            if explor_rate is None:
                self.explorer = chainerrl.explorers.ConstantEpsilonGreedy(epsilon=0.05, random_action_func=env.action_space.sample)
            else:
                self.explorer = SuccessRateEpsilonGreedy.SuccessRateEpsilonGreedy(start_epsilon=explor_rate, end_epsilon=0.05, decay_steps=50000, random_action_func=env.action_space.sample)
        else:
            self.explorer = SuccessRateEpsilonGreedy.SuccessRateEpsilonGreedy(start_epsilon=0.5, end_epsilon=0.05, decay_steps=50000, random_action_func=env.action_space.sample)
    
        #replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
        #replay_buffer = chainerrl.replay_buffer.PrioritizedReplayBuffer(capacity=10 ** 6)
        #replay_buffer = chainerrl.replay_buffer.PrioritizedEpisodicReplayBuffer(capacity=10 ** 6)
        replay_buffer_q_func = success_buffer_replay.ActionFareSamplingReplayBuffer(capacity=10 ** 6)
    
        phi = lambda x: x.astype(np.float32, copy=False)
        self.agent = chainerrl.agents.DoubleDQN(
            self.q_func, self.q_func_opt, replay_buffer_q_func, gamma, self.explorer,
            average_q_decay=0.01, average_loss_decay=0.01,
            update_interval=self.update_interval,
            target_update_interval=self.target_update_interval,
            phi=phi,
            replay_start_size=self.replay_start_size,
            minibatch_size=self.minibatch_size,
            #replay_start_size=5, minibatch_size=3, episodic_update=True, episodic_update_len=64
            )
        
        self.t = 0

    def set_success_rate(self, rate):
        self.success_rate = rate

    def obs_split_twn(self, obs):
        state32 = obs.astype(np.float32)

        twn_status = state32[0:self.n_size_twn_status]
        x_ray = state32[self.n_size_twn_status:self.n_size_twn_status+self.num_ray]
        x = x_ray.reshape(1, self.num_ray)
        eb_status = state32[self.n_size_twn_status+self.num_ray:self.n_size_twn_status+self.num_ray+self.n_size_eb_status]
        
        return twn_status, x, eb_status

    def append_history_data(self, x, is_state_terminal=False):
        self.history_data.append(x)
        if len(self.history_data) == self.history_num:
            xx = np.hstack(self.history_data)
            self.replay_buffer_hist_ana_ae.append(xx, None, 0.0, is_state_terminal=is_state_terminal)
            self.hist_ana_ae_last_out = self.hist_ana_ae.fwd(xx.reshape(1,-1))
        elif len(self.history_data) > self.history_num:
            self.history_data.pop(0)
            self.history_append_count += 1
            if self.history_append_count >= self.history_update_interval:
                self.history_append_count = 0
                xx = np.hstack(self.history_data)
                self.replay_buffer_hist_ana_ae.append(xx, None, 0.0, is_state_terminal=is_state_terminal)
                self.hist_ana_ae_last_out = self.hist_ana_ae.fwd(xx.reshape(1,-1))

        return self.hist_ana_ae_last_out
    
    def clear_history_data(self):
        self.history_append_count = 0
        self.history_data = []
        self.hist_ana_ae_last_losses = None
        self.hist_ana_ae_last_out = None

    def update_hist_ana_ae(self):
        num_sample = self.minibatch_size
        if len(self.replay_buffer_hist_ana_ae) < num_sample:
            num_sample = len(self.replay_buffer_hist_ana_ae)
        sample_obs = self.replay_buffer_hist_ana_ae.sample(num_sample)
        obs_np = np.array([elem['state'] for elem in sample_obs])
        
        self.hist_ana_ae.cleargrads()
        self.hist_ana_ae_last_losses = self.hist_ana_ae(obs_np)
        for loss in self.hist_ana_ae_last_losses:
            loss.backward()
        for opt in self.hist_ana_ae_opts:
            opt.update()



    def update_cnn_ae(self):
        sample_obs = self.replay_buffer_cnn_ae.sample(self.minibatch_size)
        obs_np = np.array([elem['state'] for elem in sample_obs])
        
        self.cnn_ae.cleargrads()
        self.cnn_ae_last_losses = self.cnn_ae(obs_np)
        for loss in self.cnn_ae_last_losses:
            loss.backward()
        for opt in self.cnn_ae_opts:
            opt.update()

    def pre_layer_output(self, obs):
        twn_status, x, eb_status = self.obs_split_twn(obs)

        with chainer.using_config("train", False):
            with chainer.no_backprop_mode():
                self.replay_buffer_cnn_ae.append(x, None, 0.0)
                
                ch, element = x.shape
                x_ray_out = self.cnn_ae.fwd(x.reshape(1,ch,element))
                x_ray_out_np = x_ray_out.data.reshape(-1)
        
                h3_c = np.hstack([twn_status, x_ray_out_np, eb_status])

                h3_h = self.append_history_data(h3_c)
                while h3_h is None:
                    h3_h = self.append_history_data(h3_c)

                h3_h_np = h3_h.data.reshape(-1)

                ans = np.hstack([twn_status, x_ray_out_np, eb_status, h3_h_np])
        
        return ans

    def act_and_train(self, obs, reward):
        """Select an action for training.

        Returns:
            ~object: action
        """
        self.t += 1

        h3_c = self.pre_layer_output(obs)

        self.explorer.set_success_rate(self.success_rate)
        
        action = self.agent.act_and_train(h3_c, reward)
        
        if (self.t % self.update_interval) == 0:
            if self.t > self.replay_start_size:
                self.update_cnn_ae()
                self.update_hist_ana_ae()
        
        return action

    def act(self, obs):
        """Select an action for evaluation.

        Returns:
            ~object: action
        """
        h3_c = self.pre_layer_output(obs)
        
        action = self.agent.act(h3_c)
        
        return action
        

    def stop_episode_and_train(self, obs, reward, done=False):
        """Observe consequences and prepare for a new episode.

        Returns:
            None
        """
        self.t += 1

        h3_c = self.pre_layer_output(obs)
        
        self.agent.stop_episode_and_train(h3_c, reward, done)
        
        if self.t > self.replay_start_size:
            self.update_cnn_ae()
            self.update_hist_ana_ae()

        self.clear_history_data()


    def stop_episode(self):
        """Prepare for a new episode.

        Returns:
            None
        """
        self.replay_buffer_cnn_ae.stop_current_episode()
        self.replay_buffer_hist_ana_ae.stop_current_episode()
        self.agent.stop_episode()

        self.clear_history_data()


    def save(self, dirname):
        """Save internal states.

        Returns:
            None
        """
        self.cnn_ae.save(os.path.join(dirname, 'cnn_ae'))
#        i = 0
#        for opt in self.cnn_ae_opts:
#            opt.save(os.path.join(dirname, 'cnn_ae_opts', '{}'.format(i)))
#            i += 1
        self.agent.save(os.path.join(dirname, 'agent'))


    def load(self, dirname):
        """Load internal states.

        Returns:
            None
        """
        self.cnn_ae.load(os.path.join(dirname, 'cnn_ae'))
#        i = 0
#        for opt in self.cnn_ae_opts:
#            opt.load(os.path.join(dirname, 'cnn_ae_opts', '{}'.format(i)))
#            i += 1
        self.agent.load(os.path.join(dirname, 'agent'))

    def get_statistics(self):
        """Get statistics of the agent.

        Returns:
            List of two-item tuples. The first item in a tuple is a str that
            represents the name of item, while the second item is a value to be
            recorded.

            Example: [('average_loss': 0), ('average_value': 1), ...]
        """
        if self.hist_ana_ae.debug_info is not None:
            ans = [(m.layer_name, di[1].data) for m, di in zip(self.hist_ana_ae.model_list, self.hist_ana_ae.debug_info)]
            #print(ans)
        else:
            ans = []
#        if self.last_losses is not None:
#            ans.extend([('cnn1_loss', self.last_losses[0].data), ('cnn2_loss', self.last_losses[1].data), ('clasify_loss', self.last_losses[2].data)])
        ans.extend(self.agent.get_statistics())
        return ans

    def get_statistics_formated_string(self):
        stat = self.get_statistics()
        stat_strings = []
        for t in stat:
            stat_strings.append('({}:{: >7.2f})'.format(t[0], t[1]))
        stat_strings.append('(explorer rate({: >6}): {:>7.3%})'.format(self.agent.t, self.explorer.compute_epsilon(self.agent.t) ) )

        ans = ','.join(stat_strings)

        return ans


def func_agent_generation(args, env, load_flag=False, explor_rate=None, load_name=None):
    
    agent = MMAgent_DDQN(args, env, load_flag, explor_rate)

    #if len(args.load) > 0:
    if load_flag:
        #agent.load(args.load)
        agent.load('agent_{}'.format(load_name))
        #logger.debug('load: {}'.format(args.load) )
        print('load: {}'.format('agent') )

    return agent

