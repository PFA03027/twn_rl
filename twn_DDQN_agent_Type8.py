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
import twn_model_base

class Qfunc_FC_TWN2_Vision(Qfunc.StateQFunction, agent.AttributeSavingMixin):
    """行動価値関数 = Q関数

    Args:
        n_size_twn_status: TWN自身の状態
        num_ray: TWNセンサーでとらえた周囲の状況
        n_size_eb_status: TWNセンサーでとらえた、EBの情報
        n_actions: 離散アクション空間
        explor_rate=0.0: 探索行動比率（現時点で未使用）
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

            print('Layer {}: in: {}  out: conv out {}  pooling out {}'.format(name, self.num_in_elements, self.num_of_conv_out_elements, self.num_of_pooling_out_elements))

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
        
        saved_attributes = ('l4','action_chain_list')
    
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
                self.l4 = links.MLP(n_in_elements, (n_in_elements*2)//3, (n_in_elements*2, int(n_in_elements*1.8), int(n_in_elements*1.5), int(n_in_elements*1.2), n_in_elements, int(n_in_elements*0.8)), nonlinearity=F.leaky_relu)
                local_action_links_list = []
                for i in range(n_actions):
                    action_links = links.MLP((n_in_elements*2)//3, 1, (n_in_elements//2,), nonlinearity=F.leaky_relu)
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
            h4 = self.l4(x)
            action_Qvalues = []
            for act_mlp in self.action_chain_list:
                qout = act_mlp(h4)
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


class MMAgent_DDQN(agent.Agent, agent.AttributeSavingMixin, twn_model_base.TWNAgentMixin):

    saved_attributes = ['cnn_ae', 'q_func', 'q_func_opt']

    def __init__(self, args, env, load_flag=False, explor_rate=None):
        super().__init__()

        self.n_size_twn_status = env.obs_size_list[0]
        self.num_ray           = env.obs_size_list[1]
        self.n_size_eb_status  = env.obs_size_list[2]

        self.update_interval = 10
        self.target_update_interval = 200
        self.replay_start_size = 1000
        self.minibatch_size = 256

        gamma = 0.99
        alpha = 0.5
        
        n_clasfy_ray = 32

#        self.q_func = Qfunc_FC_TWN2_Vision(env.obs_size_list[0], env.obs_size_list[1], env.obs_size_list[2], env.action_space.n)
        self.cnn_ae = Qfunc_FC_TWN2_Vision(self.num_ray, n_clasfy_ray)
        self.cnn_ae_opts = self.cnn_ae.gen_setup_optimizer(chainer.optimizers.Adam)
        self.replay_buffer_cnn_ae = success_buffer_replay.SuccessPrioReplayBuffer(capacity=10 ** 6)

        self.q_func = Qfunc_FC_TWN_RL(self.n_size_twn_status + n_clasfy_ray + self.n_size_eb_status, env.action_space.n)
        self.q_func_opt = chainer.optimizers.Adam(eps=1e-2)
        self.q_func_opt.setup(self.q_func)
        if load_flag:
            if explor_rate is None:
                explorer = chainerrl.explorers.ConstantEpsilonGreedy(epsilon=0.05, random_action_func=env.action_space.sample)
            else:
                explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(start_epsilon=explor_rate, end_epsilon=0.05, decay_steps=50000, random_action_func=env.action_space.sample)
        else:
            explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(start_epsilon=0.5, end_epsilon=0.05, decay_steps=50000, random_action_func=env.action_space.sample)
    
        #replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
        #replay_buffer = chainerrl.replay_buffer.PrioritizedReplayBuffer(capacity=10 ** 6)
        #replay_buffer = chainerrl.replay_buffer.PrioritizedEpisodicReplayBuffer(capacity=10 ** 6)
        replay_buffer_q_func = success_buffer_replay.ActionFareSamplingReplayBuffer(capacity=10 ** 6)
    
        phi = lambda x: x.astype(np.float32, copy=False)
        self.agent = chainerrl.agents.DoubleDQN(
            self.q_func, self.q_func_opt, replay_buffer_q_func, gamma, explorer,
            average_q_decay=0.01, average_loss_decay=0.01,
            update_interval=self.update_interval,
            target_update_interval=self.target_update_interval,
            phi=phi,
            replay_start_size=self.replay_start_size,
            minibatch_size=self.minibatch_size,
            #replay_start_size=5, minibatch_size=3, episodic_update=True, episodic_update_len=64
            )
        
        self.t = 0
        self.last_losses = None
    
    def obs_split_twn(self, obs):
        state32 = obs.astype(np.float32)

        twn_status = state32[0:self.n_size_twn_status]
        x_ray = state32[self.n_size_twn_status:self.n_size_twn_status+self.num_ray]
        x = x_ray.reshape(1, self.num_ray)
        eb_status = state32[self.n_size_twn_status+self.num_ray:self.n_size_twn_status+self.num_ray+self.n_size_eb_status]
        
        return twn_status, x, eb_status
    
    def update(self):
        sample_obs = self.replay_buffer_cnn_ae.sample(self.minibatch_size)
        obs_np = np.array([elem['state'] for elem in sample_obs])
        
        self.cnn_ae.cleargrads()
        self.last_losses = self.cnn_ae(obs_np)
        for loss in self.last_losses:
            loss.backward()
        for opt in self.cnn_ae_opts:
            opt.update()


    def act_and_train(self, obs, reward):
        """Select an action for training.

        Returns:
            ~object: action
        """
        self.t += 1
        
        twn_status, x, eb_status = self.obs_split_twn(obs)

        with chainer.using_config("train", False):
            with chainer.no_backprop_mode():
                self.replay_buffer_cnn_ae.append(x, None, reward)
                
                ch, element = x.shape
                x_ray_out = self.cnn_ae.fwd(x.reshape(1,ch,element))
                x_ray_out_np = x_ray_out.data.reshape(-1)
        
                h3_c = np.hstack([twn_status, x_ray_out_np, eb_status])
        
        action = self.agent.act_and_train(h3_c, reward)
        
        if (self.t % self.update_interval) == 0:
            if self.t > self.replay_start_size:
                self.update()
        
        return action

    def act(self, obs):
        """Select an action for evaluation.

        Returns:
            ~object: action
        """
        twn_status, x, eb_status = self.obs_split_twn(obs)
        
        with chainer.using_config("train", False):
            with chainer.no_backprop_mode():
                ch, element = x.shape
                x_ray_out = self.cnn_ae.fwd(x.reshape(1,ch,element))
                x_ray_out_np = x_ray_out.data.reshape(-1)
        
                h3_c = np.hstack([twn_status, x_ray_out_np, eb_status])
        
        action = self.agent.act(h3_c)
        
        return action
        

    def stop_episode_and_train(self, state, reward, done=False):
        """Observe consequences and prepare for a new episode.

        Returns:
            None
        """
        self.t += 1

        twn_status, x, eb_status = self.obs_split_twn(state)
        
        with chainer.using_config("train", False):
            with chainer.no_backprop_mode():
                self.replay_buffer_cnn_ae.append(x, None, reward, next_state=None, next_action=None, is_state_terminal=done)
        
                ch, element = x.shape
                x_ray_out = self.cnn_ae.fwd(x.reshape(1,ch,element))
                x_ray_out_np = x_ray_out.data.reshape(-1)
        
                h3_c = np.hstack([twn_status, x_ray_out_np, eb_status])
        
        action = self.agent.stop_episode_and_train(h3_c, reward, done)
        
        if self.t > self.replay_start_size:
            self.update()

        return action

    def stop_episode(self):
        """Prepare for a new episode.

        Returns:
            None
        """
        self.replay_buffer_cnn_ae.stop_current_episode()
        self.agent.stop_episode()

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
        ans = []
#        if self.last_losses is not None:
#            ans.extend([('cnn1_loss', self.last_losses[0].data), ('cnn2_loss', self.last_losses[1].data), ('clasify_loss', self.last_losses[2].data)])
        ans.extend(self.agent.get_statistics())
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

