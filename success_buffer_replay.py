# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 16:02:53 2019

@author: alpha
"""

import six.moves.cPickle as pickle
import heapq
import copy
import random

import chainerrl
from chainerrl.misc.collections import RandomAccessQueue


class SuccessPrioReplayBuffer(chainerrl.replay_buffer.AbstractReplayBuffer):

    def __init__(self, capacity=None):
        self.current_episode = []
        self.current_episode_R = 0.0

        self.good_episodic_memory = []
        self.good_episodic_memory_capacity = 20
        self.good_memory = RandomAccessQueue()

        self.normal_episodic_memory = []
        self.normal_episodic_memory_capacity = 50
        self.normal_memory = RandomAccessQueue()

        self.bad_episodic_memory = []
        self.bad_episodic_memory_capacity = 10
        self.bad_memory = RandomAccessQueue()

        self.capacity = capacity
        self.all_step_count = 0
        self.episode_count = 0


    def append(self, state, action, reward, next_state=None, next_action=None,
               is_state_terminal=False, **kwargs):
        experience = dict(state=state, action=action, reward=reward,
                          next_state=next_state, next_action=next_action,
                          is_state_terminal=is_state_terminal,
                          **kwargs)
        self.current_episode.append(experience)
        
        self.current_episode_R += reward
        self.all_step_count += 1

        if is_state_terminal:
            self.stop_current_episode()

    def sample(self, n):
        count_sample = 0
        ans = []
        if len(self.bad_memory) > 0:
            n_s = min((len(self.bad_memory), n//4))
            ans.extend(self.bad_memory.sample(n_s))
            count_sample += n_s

        if len(self.normal_memory) > 0:
            n_s = min((len(self.normal_memory), (n//4)*2 - count_sample))
            ans.extend(self.normal_memory.sample(n_s))
            count_sample += n_s

        if len(self.good_memory) > 0:
            n_s = min((len(self.good_memory), (n//4)*3 - count_sample))
            ans.extend(self.good_memory.sample(n_s))
            count_sample += n_s
            
        if (count_sample < n) and (len(self.current_episode) > 0):
            n_s = min((len(self.current_episode), n - count_sample))
            #ans.extend(random.sample(self.current_episode, n_s))
            ans.extend(self.current_episode[len(self.current_episode)-1-n_s:len(self.current_episode)-1])
        
        return ans

    def __len__(self):
        return self.all_step_count

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self.good_episodic_memory, self.normal_episodic_memory, self.bad_episodic_memory, self.all_step_count, self.episode_count), f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            memory = pickle.load(f)
        if isinstance(memory, tuple):
            self.good_episodic_memory, self.normal_episodic_memory, self.bad_episodic_memory, self.all_step_count, self.episode_count = memory

            self.good_memory = RandomAccessQueue()
            for e in self.good_episodic_memory:
                self.good_memory.extend(e[2])

            self.normal_memory = RandomAccessQueue()
            for e in self.normal_episodic_memory:
                self.normal_memory.extend(e[2])

            self.bad_memory = RandomAccessQueue()
            for e in self.bad_episodic_memory:
                self.bad_memory.extend(e[2])

            self.current_episode = []
            self.current_episode_R = 0.0
        else:
            print("bad replay file")

    def stop_current_episode(self):
        if self.current_episode:
            new_normal_episode = None
            if len(self.current_episode) > 1:
                if len(self.good_episodic_memory) >= self.good_episodic_memory_capacity:
                    new_normal_episode = heapq.heappushpop(self.good_episodic_memory, (copy.copy(self.current_episode_R), copy.copy(self.episode_count), self.current_episode))
                else:
                    heapq.heappush(self.good_episodic_memory, (copy.copy(self.current_episode_R), copy.copy(self.episode_count), self.current_episode))

            self.current_episode = []
            self.episode_count += 1

            new_bad_episode = None
            if new_normal_episode is not None:
                if len(self.normal_episodic_memory) >= self.normal_episodic_memory_capacity:
                    new_bad_episode = heapq.heappushpop(self.normal_episodic_memory, new_normal_episode)
                else:
                    heapq.heappush(self.normal_episodic_memory, new_normal_episode)

            if new_bad_episode is not None:
                if len(self.bad_episodic_memory) >= self.bad_episodic_memory_capacity:
                    drop_episode = heapq.heappushpop(self.bad_episodic_memory, new_bad_episode)
                    self.all_step_count -= len(drop_episode[2])
                else:
                    heapq.heappush(self.bad_episodic_memory, new_bad_episode)

            self.good_memory = RandomAccessQueue()
            for e in self.good_episodic_memory:
                self.good_memory.extend(e[2])

            self.normal_memory = RandomAccessQueue()
            for e in self.normal_episodic_memory:
                self.normal_memory.extend(e[2])

            self.bad_memory = RandomAccessQueue()
            for e in self.bad_episodic_memory:
                self.bad_memory.extend(e[2])

        assert not self.current_episode

        self.current_episode_R = 0.0

class ActionFareSamplingReplayBuffer(chainerrl.replay_buffer.AbstractReplayBuffer):

    def __init__(self, capacity=None):
        self.action_base_experience = {}
        self.action_base_experience_count = {}
        self.action_memory = {}

        self.current_episode = []

        if capacity is None:
            self.capacity = 10 ** 6
        else:
            self.capacity = capacity

        self.all_step_count = 0


    def append(self, state, action, reward, next_state=None, next_action=None,
               is_state_terminal=False, **kwargs):
        experience = dict(state=state, action=action, reward=reward,
                          next_state=next_state, next_action=next_action,
                          is_state_terminal=is_state_terminal,
                          **kwargs)
        
        self.current_episode.append(experience)

        if (action in self.action_base_experience) == False:
            self.action_base_experience[action] = []
            self.action_base_experience_count[action] = 0
        
        self.action_base_experience[action].append(experience)
        self.action_base_experience_count[action] += 1
        
        self.all_step_count += 1
        
        if self.all_step_count > self.capacity:
            max_count = 0
            max_action = None
            for ma, mc in self.action_base_experience_count.items():
                if mc > max_count:
                    max_count = mc
                    max_action = ma
            if max_action is not None:
                self.action_base_experience[max_action].pop(0)
                self.all_step_count -= 1
                self.action_base_experience_count[max_action] -= 1

    def sample(self, n):
        count_sample = 0
        ans = []
        es = n//(len(self.action_memory.keys())+1)
        if es < 1:
            es = 1
        
        for ram in self.action_memory.keys():
            n_s = min((len(self.action_memory[ram]), es))
            if (count_sample+n_s) > n:
                n_s = n - count_sample
                if n_s <= 0:
                    break
                ans.extend(self.action_memory[ram].sample(n_s))
                count_sample += n_s
                break
            else:
                ans.extend(self.action_memory[ram].sample(n_s))
                count_sample += n_s

        if count_sample < n:
            n_s = min((len(self.current_episode), n - count_sample))
            #ans.extend(random.sample(self.current_episode, n_s))
            ans.extend(self.current_episode[len(self.current_episode)-1-n_s:len(self.current_episode)-1])

        return ans

    def __len__(self):
        return self.all_step_count

    def save(self, filename):
        pass

    def load(self, filename):
        pass

    def stop_current_episode(self):
        for ac in self.action_base_experience.keys():
            self.action_memory[ac] = RandomAccessQueue()
            self.action_memory[ac].extend(self.action_base_experience[ac])
            
        if self.current_episode:
            self.current_episode = []


