# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 20:34:08 2018

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
from chainerrl.action_value import DiscreteActionValue

import success_buffer_replay

import twn_DDQN_agent_Type1
import twn_DDQN_agent_Type2


class Evaluation_Cmd(Enum):
    '''
    評価指示するID
    '''
    INVALID = -1
    START_EVALUATION = 1
    START_EVALUATION_FORCE = 2
    FINISH = 3

def func_traning(args, mq, env_name, func_agent_generation):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    env = gym.make(env_name)
    
    max_number_of_steps = 3000  #総試行回数
    num_episodes = 500  #総試行回数
    episode = 0
    
    load_flag = False
    if len(args.load) > 0:
        load_flag = True
    agent = func_agent_generation(args, env, load_flag)
    

#    for episode in range(num_episodes):  #試行数分繰り返す
    while env.get_eb_count < num_episodes:
        episode += 1
        observation = env.reset()
        done = False
        reward = 0
        R = 0
        for t in range(max_number_of_steps):  #1試行のループ
            action = agent.act_and_train(observation, reward)
            reward = 0
            observation, reward, done, info = env.step(action)
            R += reward
            #print('reward:', reward)

            if (t%10==0) or done:
                tmp_add_info = None
                if isinstance(agent, chainerrl.agents.DoubleDQN):
                    av_data = agent.q_function.model.debug_info[0].data
                    tmp_add_info = [av_data.reshape(1,-1)]
                elif isinstance(agent, twn_DDQN_agent_Type2.MMAgent_DDQN):
                    av_data = agent.agent.q_function.model.debug_info[0].data
                    tmp_add_info = [av_data.reshape(1,-1)]
                    if agent.cnn_ae.debug_info is not None:
                        #print('aaa--> ', agent.cnn_ae.debug_info[0][3].data.shape)
                        #print('aaa--> ', agent.cnn_ae.debug_info[0][2].data.shape)
                        tmp_add_info.extend([agent.cnn_ae.debug_info[0][3][0,:,:].reshape(-1), agent.cnn_ae.debug_info[0][2][0,:,:].reshape(-1)])

                env.render(add_info=tmp_add_info)
                stat = agent.get_statistics()
#                logger.info('episode: {}  turn: {} EB: {} R: {}  statistics [(average_q: {}), (average_loss: {})]  TWN enagy: {}'.format(episode, t, env.get_eb_count, R, stat[0][1], stat[1][1], env.twn.enagy))
                logger.info('episode: {}  turn: {} EB: {} R: {}  statistics [{}]  TWN enagy: {}'.format(episode, t, env.get_eb_count, R, stat, env.twn.enagy))

            if done:
                break
        if done == False:
            reward += 10.0
        agent.stop_episode_and_train(observation, reward, done)
        stat = agent.get_statistics()
        logger.info('episode: {}  R: {}  statistics [(average_q: {}), (average_loss: {})]'.format(episode, R, stat[0][1], stat[1][1]))
        agent.save('agent')
        logger.info('Stored agent {}'.format('agent'))
        mq.put(Evaluation_Cmd.START_EVALUATION)

    logger.info('Finish training.')

    env.finish()


def func_demo(args, mq, env_name, func_agent_generation):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    env = gym.make(env_name)
    

    if mq == None:
        max_number_of_steps = 3000  #総試行回数
        
        load_flag = False
        if len(args.load) > 0:
            load_flag = True
        agent = func_agent_generation(args, env, load_flag)
        env.get_eb_count = 400
    
        for episode in range(args.eval_n_runs*10):  #試行数分繰り返す
            observation = env.reset()
            done = False
            reward = 0
            R = 0
            for t in range(max_number_of_steps):  #1試行のループ
                action = agent.act(observation)
                observation, reward, done, info = env.step(action)
                R += reward
                #print('reward:', reward)
    
                if (t%5==0) or done:
                    if isinstance(agent, chainerrl.agents.DoubleDQN):
                        av_data = agent.q_function.model.debug_info[0].data
                    elif isinstance(agent, twn_DDQN_agent_Type2.MMAgent_DDQN):
                        av_data = agent.agent.q_function.model.debug_info[0].data
                    env.render(add_info=[av_data.reshape(1,-1)])
                    stat = agent.get_statistics()
                    logger.info('episode: {}  turn: {} EB: {} R: {}  statistics [{}]  TWN enagy: {}'.format(episode, t, env.get_eb_count, R, stat, env.twn.enagy))
    
                if done:
                    break
            agent.stop_episode()
            stat = agent.get_statistics()
            logger.info('episode: {}  R: {}  statistics [(average_q: {}), (average_loss: {})]'.format(episode, R, stat[0][1], stat[1][1]))
        logger.info('Finish evaluation.')

    else:
        max_number_of_steps = 100  #総試行回数
                
        cont_flag = True
        while cont_flag:
            msg = mq.get()
            print('Evaluation command: {}'.format(msg))
            logger.info('Evaluation command: {}'.format(msg))
    
            if msg == Evaluation_Cmd.START_EVALUATION or msg == Evaluation_Cmd.START_EVALUATION_FORCE:
                if mq.empty() == True or msg == Evaluation_Cmd.START_EVALUATION_FORCE:
                    agent = func_agent_generation(args, env, load_flag=True)
                
                    for episode in range(args.eval_n_runs):  #試行数分繰り返す
                        observation = env.reset()
                        done = False
                        reward = 0
                        R = 0
                        for t in range(max_number_of_steps):  #1試行のループ
                            action = agent.act(observation)
                            observation, reward, done, info = env.step(action)
                            R += reward
                            #print('reward:', reward)
                
                            if t%10==0:
                                if isinstance(agent, chainerrl.agents.DoubleDQN):
                                    av_data = agent.q_function.model.debug_info[0].data
                                elif isinstance(agent, twn_DDQN_agent_Type2.MMAgent_DDQN):
                                    av_data = agent.agent.q_function.model.debug_info[0].data
                                env.render(add_info=[av_data.reshape(1,-1)])
                                stat = agent.get_statistics()
                                logger.info('episode: {}  turn: {}  R: {}  statistics [(average_q: {}), (average_loss: {})]  TWN enagy: {}'.format(episode, t, R, stat[0][1], stat[1][1], env.twn.enagy))
                
                            if done:
                                break
                        agent.stop_episode()
                        stat = agent.get_statistics()
                        logger.info('episode: {}  R: {}  statistics [(average_q: {}), (average_loss: {})]'.format(episode, R, stat[0][1], stat[1][1]))
                    logger.info('Finish evaluation.')
                    print('Finish evaluation.')
            elif msg == Evaluation_Cmd.FINISH:
                cont_flag = False
                logger.info('End evaluation.')
            else:
                logger.info('Invalid evaluation command.')

    env.finish()

if __name__ == '__main__':
    mp.freeze_support()
    
#    logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.CRITICAL)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--env', type=str, default='twm_box_garden-v0')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--final-exploration-steps',
                        type=int, default=10 ** 6)
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--steps', type=int, default=10 ** 7)
    parser.add_argument('--n-hidden-channels', type=int, default=300)
    parser.add_argument('--n-hidden-layers', type=int, default=3)
    parser.add_argument('--replay-start-size', type=int, default=5000)
    parser.add_argument('--n-update-times', type=int, default=1)
    parser.add_argument('--target-update-interval',
                        type=int, default=1)
    parser.add_argument('--target-update-method',
                        type=str, default='soft', choices=['hard', 'soft'])
    parser.add_argument('--soft-update-tau', type=float, default=1e-2)
    parser.add_argument('--update-interval', type=int, default=4)
    parser.add_argument('--eval-n-runs', type=int, default=8)
    parser.add_argument('--eval-interval', type=int, default=10 ** 5)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--minibatch-size', type=int, default=200)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--use-bn', action='store_true', default=False)
    parser.add_argument('--monitor', action='store_true')
    parser.add_argument('--reward-scale-factor', type=float, default=1e-2)
    args = parser.parse_args()

#    args.outdir = chainerrl.experiments.prepare_output_dir(args, args.outdir, argv=sys.argv)
#    print('Output files are saved in {}'.format(args.outdir))
    
    env_name = 'twm_box_garden-v1'
    
#    func_agent_generation = twn_DDQN_agent_Type1.func_agent_generation
    func_agent_generation = twn_DDQN_agent_Type2.func_agent_generation
    
    if args.demo:
        func_demo(args, None, env_name, func_agent_generation)

#        mq_to_demo = mp.Queue()
#        demo_process = mp.Process(target=func_demo, args=(args, mq_to_demo, env_name,))
#        demo_process.start()
#
#        mq_to_demo.put(Evaluation_Cmd.START_EVALUATION_FORCE)
#        mq_to_demo.put(Evaluation_Cmd.FINISH)
#        
#        demo_process.join()


    else:
        mq_to_demo = mp.Queue()

        demo_process = mp.Process(target=func_demo, args=(args, mq_to_demo, env_name, func_agent_generation))
        demo_process.start()

        func_traning(args, mq_to_demo, env_name, func_agent_generation)
        mq_to_demo.put(Evaluation_Cmd.FINISH)
        
        demo_process.join()


    logger.info('Exit')