# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 00:46:05 2018

@author: alpha
"""

import numpy as np
import math
from enum import Enum

import logging

#import matplotlib.pyplot as plt

import os
import copy
import multiprocessing as mp
import time
import threading
import random

import gym
from gym import spaces
from gym.utils import seeding

#import id_generator as idg
from . import BoxGarden
from . import BoxGarden_draw
from . import bg_obj
from . import two_wheel_mover as twn
from . import drawing_trace2 as dt2


class my_car_obj(bg_obj.circle_object):
    num_of_ray = 180
    radius = 0.3
    
    def __init__(self, pos=np.zeros(2, dtype=np.float32), rot_theata=0.0):
        '''
        第1引数  2つの要素を持つnumpyの配列。円の中心の座標
        '''
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.WARNING)

        super().__init__(pos, r=my_car_obj.radius)
        self.rot_theata = rot_theata
        
        self.ray = np.empty((2,my_car_obj.num_of_ray), dtype=np.float32)
        for i in range(my_car_obj.num_of_ray):
            self.ray[0,i] = math.cos(math.pi/(my_car_obj.num_of_ray/2)*(-my_car_obj.num_of_ray/2+i))
            self.ray[1,i] = math.sin(math.pi/(my_car_obj.num_of_ray/2)*(-my_car_obj.num_of_ray/2+i))

        self.target_bg_draw = None
        

    class my_car_msg_ID(Enum):
        '''
        自車オブジェクトの描画用オブジェクトへのメッセージID
        '''
        INVALID = -1
        RESTART = 1
        UPDATE_POS_ATTRS = 2

    class my_car_msg(BoxGarden_draw.bg_message_base):
        '''
        自車オブジェクトの描画用オブジェクトへのメッセージクラスの定義(クラス内クラス定義)
        '''
        def __init__(self, target_id):
            super().__init__(mid=BoxGarden_draw.bg_msg_ID.COMMON, target_id=target_id)
            self.my_car_cmd = my_car_obj.my_car_msg_ID.INVALID
        
        def restart_pos(self, pos=np.zeros(2, dtype=np.float32), rot_theata=0.0):
            self.my_car_cmd = my_car_obj.my_car_msg_ID.RESTART
            self.new_pos = pos.reshape(-1)
            self.new_rot_theata = rot_theata

        def update_pos_attrs(self, pos, rot_theata, attrs):
            self.my_car_cmd = my_car_obj.my_car_msg_ID.UPDATE_POS_ATTRS
            self.new_pos = pos.reshape(-1)
            self.new_rot_theata = rot_theata
            self.attrs = attrs
            # print("update_pos_attrs: self.new_pos: ", self.new_pos)
            # print("update_pos_attrs: self.new_rot_theata: ", self.new_rot_theata)

    class my_car_draw(BoxGarden_draw.bg_object_draw):
        '''
        自車オブジェクトを表す箱庭オブジェクトの描画用オブジェクト(クラス内クラス定義)
        '''
        def __init__(self, pos=np.zeros(2, dtype=np.float32), ptidx=0):
            '''
            第1引数  2つの要素を持つnumpyの配列。円の中心の座標
            第2引数  半径
            '''
            super().__init__()
            self.pos = pos
            self.plot_target_idx = ptidx
            
            self.attrs = []
            self.update_count = 0

            print("my_car_draw.__init__: ID: ", self.id)

        
        def process_add(self, msg, bg_draw):
            ax = bg_draw.scn.subplot_list[self.plot_target_idx].ax
            
            self.my_do = dt2.drawobj_empty()
            self.my_do.move = self.pos
            dt2.drawobj_circle(ax, 0.0, 0.0, my_car_obj.radius, attr='r-', parent=self.my_do)
            dt2.drawobj_triangle(ax, 0.0, 0.0, -math.sqrt(0.75), 0.5, -math.sqrt(0.75), -0.5, parent=self.my_do)
            
            bg_draw.scn.subplot_list[self.plot_target_idx].append_drawobj(self.my_do)

            ax2 = bg_draw.scn.subplot_list[self.plot_target_idx+1].ax
    
            self.line2 = dt2.drawobj_poly(ax2, attr='r-')
            self.line2_b1 = dt2.drawobj_poly(ax2, attr='g-')
            self.line2_b2 = dt2.drawobj_poly(ax2, attr='b-')
            for i in range(my_car_obj.num_of_ray):
                self.line2.append( -180 + i*360.0/my_car_obj.num_of_ray, 0.0 )
                self.line2_b1.append( -180 + i*360.0/my_car_obj.num_of_ray, 0.0 )
                self.line2_b2.append( -180 + i*360.0/my_car_obj.num_of_ray, 0.0 )

            bg_draw.scn.subplot_list[self.plot_target_idx+1].append_drawobj(self.line2)
            bg_draw.scn.subplot_list[self.plot_target_idx+1].append_drawobj(self.line2_b1)
            bg_draw.scn.subplot_list[self.plot_target_idx+1].append_drawobj(self.line2_b2)

            ax5 = bg_draw.scn.subplot_list[self.plot_target_idx+2].ax
            self.cum_reward = dt2.drawobj_poly(ax5, attr='b-')
            self.cum_reward.append( 0.0, 0.0 )
            bg_draw.scn.subplot_list[self.plot_target_idx+2].append_drawobj(self.cum_reward)

            ax4 = bg_draw.scn.subplot_list[self.plot_target_idx+3].ax
            self.eb_direction = dt2.drawobj_poly(ax4, attr='r-')
            self.eb_direction.append( 0.0, 0.0 )
            self.eb_direction.append( 0.0, 0.0 )
            bg_draw.scn.subplot_list[self.plot_target_idx+3].append_drawobj(self.eb_direction)

            ax6 = bg_draw.scn.subplot_list[self.plot_target_idx+4].ax
            self.twn_wh_v_left = dt2.drawobj_poly(ax6, attr='r-')
            self.twn_wh_v_left.append( -0.5, 0.0 )
            self.twn_wh_v_left.append( -0.5, 0.0 )
            bg_draw.scn.subplot_list[self.plot_target_idx+4].append_drawobj(self.twn_wh_v_left)
            self.twn_wh_v_right = dt2.drawobj_poly(ax6, attr='r-')
            self.twn_wh_v_right.append(  0.5, 0.0 )
            self.twn_wh_v_right.append(  0.5, 0.0 )
            bg_draw.scn.subplot_list[self.plot_target_idx+4].append_drawobj(self.twn_wh_v_right)

            ax7 = bg_draw.scn.subplot_list[self.plot_target_idx+5].ax
            self.add_info = dt2.drawobj_poly(ax7, attr='b-')
            self.add_info.append( 0.0, 0.0 )
            bg_draw.scn.subplot_list[self.plot_target_idx+5].append_drawobj(self.add_info)

        def process_msg(self, msg, bg_draw):
            '''
            メッセージ処理を実行する
            '''
            if msg.my_car_cmd is not None:
                if msg.my_car_cmd == my_car_obj.my_car_msg_ID.RESTART:
                    self.my_do.move = msg.new_pos
                    self.my_do.rot_theata = msg.new_rot_theata
                    self.attrs = []
                    for i in range(my_car_obj.num_of_ray):
                        self.line2.y[i] = 0.0
                        self.line2_b1.y[i] = 0.0
                        self.line2_b2.y[i] = 0.0
                    self.eb_direction.clear()
                    self.eb_direction.append( 0.0, 0.0 )
                    self.eb_direction.append( 0.0, 0.0 )
                    self.cum_reward.clear()
                    self.cum_reward.append( 0.0, 0.0 )
                    self.update_count = 0
                    self.add_info.clear()
                    self.add_info.append( 0.0, 0.0 )
                elif msg.my_car_cmd == my_car_obj.my_car_msg_ID.UPDATE_POS_ATTRS:
                    self.my_do.move = msg.new_pos
                    self.my_do.rot_theata = msg.new_rot_theata
                    if len(msg.attrs) != 0:
                        self.attrs = msg.attrs
                        for i in range(my_car_obj.num_of_ray):
                            self.line2.y[i] = self.attrs[1][i]
                        vv_d = my_car_obj.get_rot_mat( math.pi/2.0 ).dot(np.array((self.attrs[2][0], self.attrs[2][1]), dtype=np.float32).reshape(2,1))
                        #vv_d = np.array((self.attrs[2][0], self.attrs[2][1]), dtype=np.float32).reshape(2,1)
                        self.eb_direction.x[1] = vv_d[0,0] * self.attrs[2][2]
                        self.eb_direction.y[1] = vv_d[1,0] * self.attrs[2][2]
                        
                        self.twn_wh_v_left.y[1] = self.attrs[0][0]
                        self.twn_wh_v_right.y[1] = self.attrs[0][1]

                        self.update_count += 1
                        self.cum_reward.append( self.update_count, self.attrs[3] )
                        
                        if self.attrs[4] is not None:
                            nrow, ncol = self.attrs[4][0].shape
                            self.add_info.clear()
                            for col_idx in range(ncol):
                                self.add_info.append( col_idx, self.attrs[4][0][0, col_idx] )
                            if len(self.attrs[4]) >= 3:
                                for i in range(my_car_obj.num_of_ray):
                                    self.line2_b1.y[i] = self.attrs[4][1].data[i]
                                    self.line2_b2.y[i] = self.attrs[4][2].data[i]
                                

                            #print("process_msg: self.line2.y[i]: ", self.line2.y[i])
                    # print("process_msg: self.move: ", self.my_do.move)
                    # print("process_msg: self.rot_theata: ", self.my_do.rot_theata)

        def update_subplot(self, data):
            pass
    
    def send_drawobj(self, bg_draw):
        '''
        描画用オブジェクトをBoxGarden_drawに送る
        '''
        self.target_bg_draw = bg_draw

        obj = my_car_obj.my_car_draw(self.pos)

        self.do_id = obj.id     # 相手先のidを保持しておく
        mes = BoxGarden_draw.bg_message_base()
        mes.command_ADD(obj)
        self.target_bg_draw.send_msg(mes)

    def restart_pos(self, pos=np.zeros(2, dtype=np.float32), rot_theata=0.0):
        self.pos = pos.reshape(2,1).astype(np.float32)
        self.rot_theata = rot_theata
        mes = my_car_obj.my_car_msg(self.do_id)
        mes.restart_pos(pos,rot_theata)
        if self.target_bg_draw is not None:
            self.target_bg_draw.send_msg(mes)

    def update_pos_attrs(self, pos, rot_theata, attrs):
        self.pos = pos.reshape(2,1).astype(np.float32)
        self.rot_theata = rot_theata
        mes = my_car_obj.my_car_msg(self.do_id)
        mes.update_pos_attrs(pos,rot_theata,attrs)
        if self.target_bg_draw is not None:
            self.target_bg_draw.send_msg(mes)

    @classmethod
    def senser_minmax(cls):
        return np.zeros(cls.num_of_ray, dtype=np.float32), np.ones(cls.num_of_ray, dtype=np.float32)
        #return np.zeros(cls.num_of_ray, dtype=np.float32).reshape(1,-1), np.ones(cls.num_of_ray, dtype=np.float32).reshape(1,-1)
    
    @classmethod
    def get_rot_mat(cls, r):
        return np.matrix( (( math.cos(r), -math.sin(r)),
                           ( math.sin(r),  math.cos(r)) ), dtype=np.float32 )
        
    def get_rotated_ray(self):
        rot_mat = my_car_obj.get_rot_mat(self.rot_theata)
        return rot_mat.dot(self.ray)



class twn_BoxGarden_draw(BoxGarden_draw.BoxGarden_draw):
    def __init__(self):
        super().__init__([2,3])

    def start_screen(self):
        def reward_screen_post_update(ax):
            #print("Whoo")
            ax.set_xlim(left=0.0, auto=True)
            ax.set_ylim(auto=True)

        super().start_screen()

        self.scn.subplot_list[0].ax.set_xlim(-17, 17)
        self.scn.subplot_list[0].ax.set_ylim(-17, 17)
        self.scn.subplot_list[0].ax.set_xlabel('x')
        self.scn.subplot_list[0].ax.set_ylabel('y')
        self.scn.subplot_list[0].ax.grid()

        self.scn.subplot_list[1].ax.set_xlim(-180, 180)
        self.scn.subplot_list[1].ax.set_ylim(0, 1)
        self.scn.subplot_list[1].ax.set_xlabel('x')
        self.scn.subplot_list[1].ax.set_ylabel('y')

        self.scn.subplot_list[2].ax.set_xlim(0.0, 300.0)
        self.scn.subplot_list[2].ax.set_ylim(-300.0, 100.0)
        self.scn.subplot_list[2].ax.set_xlabel('update count')
        self.scn.subplot_list[2].ax.set_ylabel('cumulative value of reward')
        self.scn.subplot_list[2].post_update_func = reward_screen_post_update

        self.scn.subplot_list[3].ax.set_xlim(-1, 1)
        self.scn.subplot_list[3].ax.set_ylim(-1, 1)
        self.scn.subplot_list[3].ax.set_xlabel('L<-->R')
        self.scn.subplot_list[3].ax.set_ylabel('B<-->F')

        self.scn.subplot_list[4].ax.set_xlim(-1, 1)
        self.scn.subplot_list[4].ax.set_ylim(-1, 1)
        self.scn.subplot_list[4].ax.set_xlabel('L<-->R')
        self.scn.subplot_list[4].ax.set_ylabel('Velocity')

        self.scn.subplot_list[5].ax.set_xlim(0, 25)
        self.scn.subplot_list[5].ax.set_ylim(-150, 150)
        self.scn.subplot_list[5].ax.set_xlabel('action index')
        self.scn.subplot_list[5].ax.set_ylabel('Q value')


class training_base:
    def __init__(self):
        self.success_count = 0
        self.try_count = 0
    
    def set_end_status(self, success_fail):
        ''' If success_fail is True, success_count is increased. '''
        if success_fail:
            self.success_count += 1

    def check_end_status(self, obs, reward_map, status_get_eb_flag):
        raise NotImplementedError()

    def get_new_eb_pos(self):
        raise NotImplementedError()

class training_type1(training_base):
    def check_end_status(self, obs, reward_map, status_get_eb_flag):
        self.set_end_status(status_get_eb_flag)

    def get_new_eb_pos(self):
        self.try_count += 1
        dist = 1.5 + self.try_count % (self.success_count%12+1)
        if dist > TWN_BoxGardenEnv.wall_area_range * 0.8:
            dist = TWN_BoxGardenEnv.wall_area_range * 0.8
        return np.array([0.0, dist]).reshape(2,1)

class training_type2(training_base):
    def check_end_status(self, obs, reward_map, status_get_eb_flag):
        self.set_end_status(status_get_eb_flag)

    def get_new_eb_pos(self):
        self.try_count += 1
        dist = 3.0 + self.try_count % (self.success_count%10+1)
        if dist > TWN_BoxGardenEnv.wall_area_range * 0.8:
            dist = TWN_BoxGardenEnv.wall_area_range * 0.8
        return my_car_obj.get_rot_mat( math.pi/180.0*(random.uniform(-9.0,9.0)) ).dot(np.array([0.0, random.uniform(3.0, dist)])).reshape(2,1)

class training_type3(training_base):
    def check_end_status(self, obs, reward_map, status_get_eb_flag):
        self.set_end_status(status_get_eb_flag)

    def get_new_eb_pos(self):
        self.try_count += 1
        dist = 3.0 + self.try_count % (self.success_count%10+1)
        if dist > TWN_BoxGardenEnv.wall_area_range * 0.8:
            dist = TWN_BoxGardenEnv.wall_area_range * 0.8
        angle_range = 10.0 + self.success_count
        if angle_range > 80.0:
            angle_range = 80.0
        return my_car_obj.get_rot_mat( math.pi/180.0*(random.uniform(-angle_range,angle_range)) ).dot(np.array([0.0, random.uniform(3.0, dist)])).reshape(2,1)

class training_type4(training_base):
    def check_end_status(self, obs, reward_map, status_get_eb_flag):
        self.set_end_status(status_get_eb_flag)

    def get_new_eb_pos(self):
        self.try_count += 1
        dist = 3.0 + self.try_count % (self.success_count%12+1)
        if dist > TWN_BoxGardenEnv.wall_area_range:
            dist = TWN_BoxGardenEnv.wall_area_range
        return np.random.uniform(-dist, dist, 2).reshape(2,1)



class TWN_BoxGardenEnv(gym.Env):
    eb_enagy = 300
    wall_area_range = 15.0
    
    def __init__(self, discrate_action=True, history_size=1, renderable=True):
        self.logger = logging.getLogger(__name__)
        #self.logger.setLevel(logging.WARNING)
        self.logger.setLevel(logging.DEBUG)
        self.logger.info("TWN_BoxGardenEnv instance is generated!!!!!!!!!!!!!!!!!!!!!!!!!!")

        self.twn_bg = BoxGarden.BoxGarden()
        self.wall_obj = bg_obj.circle_wall(r=TWN_BoxGardenEnv.wall_area_range)  # 外壁
        self.cylinders = [  # お邪魔物体
                bg_obj.circle_object(pos=np.array([ 5.0,  5.0]), r=1.0),
                bg_obj.circle_object(pos=np.array([-5.0,  5.0]), r=1.0),
                bg_obj.circle_object(pos=np.array([-5.0, -5.0]), r=1.0),
                bg_obj.circle_object(pos=np.array([ 5.0, -5.0]), r=1.0)
                ]

        self.twn = twn.TwoWheelMover_SetWheelVec()
        self.my_car = my_car_obj(pos=self.twn.pos, rot_theata=self.twn.GetRotTheata())
        
        self.eb = bg_obj.circle_object(pos=np.array([2.0, 2.0]), r=0.05)    # 目標の物体
        #self.reset_eb_pos()

        self.twn_bg.append_bg_obj(self.wall_obj)
        self.twn_bg.append_bg_obj(self.eb)
        for cy in self.cylinders:
            self.twn_bg.append_bg_obj(cy)
        self.twn_bg.append_bg_obj(self.my_car)
        
        twn.TwoWheelMover.max_wheel_vec =  0.5
        twn.TwoWheelMover.min_wheel_vec = -0.5
        twn.TwoWheelMover.max_wheel_acc =  0.05
        twn.TwoWheelMover.min_wheel_acc = -0.05
        
        self.delta_t = 0.02    # 20msecで、更新
        self.num_delta_t_per_action = 10
        twn.TwoWheelMover.delta_t = self.delta_t
        
        if renderable:
            self.twn_bg_draw = twn_BoxGarden_draw()
            self.bg_draw_boot = BoxGarden_draw.BoxGarden_draw_boot(self.twn_bg_draw)
            self.wall_obj.send_drawobj(self.twn_bg_draw)
            self.eb.send_drawobj(self.twn_bg_draw)
            for cy in self.cylinders:
                cy.send_drawobj(self.twn_bg_draw)
            self.my_car.send_drawobj(self.twn_bg_draw)
        else:
            self.twn_bg_draw = None
        
        self.action_space_type = discrate_action    # True = 離散タイプ
        if self.action_space_type:
            self.logger.info("action Discrete")
            self.action_space_divide = 2
            self.action_space_divide_plus1 = self.action_space_divide + 1
            self.action_space = spaces.Discrete(self.action_space_divide_plus1 * self.action_space_divide_plus1)
            self.action_space_low  = np.array([twn.TwoWheelMover.min_wheel_vec, twn.TwoWheelMover.min_wheel_vec])
            self.action_space_high = np.array([twn.TwoWheelMover.max_wheel_vec, twn.TwoWheelMover.max_wheel_vec])
        else:
            self.action_space = spaces.Box(np.array([twn.TwoWheelMover.min_wheel_vec, twn.TwoWheelMover.min_wheel_vec]), np.array([twn.TwoWheelMover.max_wheel_vec, twn.TwoWheelMover.max_wheel_vec]))
            self.logger.info("action Box: (low, {}), (high, {}) ".format(self.action_space.low, self.action_space.high))
        
        # 観測データのヒストリーサイズ
        self.obs_hist_num = history_size
        
        # TWNの状態
        rot_speed_minmax = self.twn.GetRotateSpeedMinMax()
        low1 = np.array([twn.TwoWheelMover.min_wheel_vec, twn.TwoWheelMover.min_wheel_acc, rot_speed_minmax[0], 0])
        high1 = np.array([twn.TwoWheelMover.max_wheel_vec, twn.TwoWheelMover.max_wheel_acc, rot_speed_minmax[1], twn.TwoWheelMover.max_enagy])

        # 視線センサーの状態
        low2, high2 = my_car_obj.senser_minmax()

        # 餌のある方角と距離に相当する情報
        low3 = np.array([-1.0, -1.0, 0.0])
        high3 = np.array([1.0, 1.0, 1.0])

        # 観測データの最大、最小情報の結合
        unit_low = [low1, low2, low3]
        list_low =[]
        unit_high = [high1, high2, high3]
        list_high = []
        for i in range(self.obs_hist_num):
            list_low.extend(unit_low)
            list_high.extend(unit_high)
        low  = np.hstack(list_low)
        high = np.hstack(list_high)
        self.observation_space = spaces.Box(low, high)
        self.obs_size_list = [len(low1), len(low2), len(low3)]
        self.logger.info("observation Box: (low, {}), (high, {}) ".format(self.observation_space.low, self.observation_space.high))
        
        self.training_step = 0
        self.try_count = 0
        
        self.ob1 = None
        self.ob2 = None
        self.ob3 = None
        self.old_ob1 = None
        self.old_ob2 = None
        self.old_ob3 = None
        self.old_dist = None

        self.cumulative_value_of_reward = 0.0
        self.get_eb_count = 0
        self.collision_count = 0
        
        self.reward_scalor = 1.0
        
        self.trainers = [training_type1(), training_type2(), training_type3(), training_type4()]
        self.current_trainer = self.trainers[0]

    def collision_my_car(self):
        ans = None

        if self.collision_circle2wall(self.twn.pos, self.my_car.radius, self.wall_obj.pos, self.wall_obj.radius):
            ans = 'wall'
        elif self.collision_circle2circle(self.twn.pos, self.my_car.radius, self.eb.pos, self.eb.radius):
            ob3_angle = math.atan2(self.ob3[1], self.ob3[0])
            # ob3_angle = math.atan2( -self.ob3[0], self.ob3[1] )
            if (-math.pi/180*10 < ob3_angle) & (ob3_angle < math.pi/180*10):
                ans = 'eb_IN'  # 正面からの場合は、OK
            else:
                ans = 'eb'  # 後ろからEBにぶつかったら、衝突とする。
        else:
            for cy in self.cylinders:
                if self.collision_circle2circle(self.twn.pos, self.my_car.radius, cy.pos, cy.radius):
                    ans = 'cylinder'
#                    ans = True

        return ans
    
    def collision_eb(self, new_pos):
        ans = False

        if self.collision_circle2wall(new_pos, self.eb.radius, self.wall_obj.pos, self.wall_obj.radius):
            ans = True
        elif self.collision_circle2circle(new_pos, self.eb.radius, self.twn.pos, self.my_car.radius):
            ans = True
        else:
            for cy in self.cylinders:
                if self.collision_circle2circle(new_pos, self.eb.radius, cy.pos, cy.radius):
                    ans = True

        return ans
    
    def step_next_state(self, action):
        if self.action_space_type:    # True = 離散タイプ
            wv_tmp = np.array([action // self.action_space_divide_plus1, action % self.action_space_divide_plus1]) * ((self.action_space_high - self.action_space_low) /self.action_space_divide) + self.action_space_low
            left_wv  = wv_tmp[0]
            right_wv = wv_tmp[1]
        else:
            left_wv  = action[0]
            right_wv = action[1]
        #self.logger.info( 'wheel velocty: left: {}  right: {}'.format(left_wv, right_wv) )
        #print( 'action: {} -> wheel velocty: left: {}  right: {}'.format(action, left_wv, right_wv) )

        # Set-up action
        self.twn.SetOperationParam(left_wv, right_wv)
        
        # 状態の更新
        eb_in_flag = False
        collision_count = 0
        for i in range(self.num_delta_t_per_action):
            old_pos = copy.copy(self.twn.pos)
            self.twn.MoveNext()
            col = self.collision_my_car()
            if col is not None:
                if col == 'eb_IN':
                    if eb_in_flag:
                        self.twn.pos = old_pos
                    else:
                        eb_in_flag = True
                else:
                    self.twn.pos = old_pos
                    collision_count += 1
        
        return collision_count, eb_in_flag
    
    def gather_observation(self):
        # 自身の情報を集める
        rot_speed = self.twn.GetRotateSpeed()
        self.ob1 = np.array([self.twn.wheel_vec[0], self.twn.wheel_vec[1], rot_speed, self.twn.enagy], dtype=np.float32)

        # 視線交差確認で距離を求め、ob2の更新
        self.ob2 = np.empty(my_car_obj.num_of_ray, dtype=np.float32)
        rotated_ray = self.my_car.get_rotated_ray()
        ret = self.twn_bg.culc_nearest_distance_to_cross_Nray(copy.copy(self.twn.pos), rotated_ray, ignore_id=self.my_car.id)
        for i in range(my_car_obj.num_of_ray):
            if ret[0][i] == BoxGarden.bg_obj_RC.NOT_CROSSED:
                self.ob2[i] = 0.0
            elif ret[0][i] == BoxGarden.bg_obj_RC.IN_OBJECT:
                self.ob2[i] = 1.0
            elif ret[0][i] == BoxGarden.bg_obj_RC.CROSSED:
                if ret[1][i] <= self.my_car.radius:
                    self.ob2[i] = 1.0
                else:
                    self.ob2[i] = abs(1.0/(1.0+ret[1][i]-self.my_car.radius)+random.normalvariate(0.01, 0.005))   # 大体1%の計測誤差

        
        # エネルギー体までの情報を集める
        ob3_a = self.eb.pos - self.twn.pos
        ob3_b = np.linalg.norm(ob3_a)
        ob3_c = ob3_a / ob3_b
        ob3_c = my_car_obj.get_rot_mat( - self.twn.GetRotTheata() ).dot(ob3_c)
        ob3_d = 1.0/(1.0+ob3_b-self.my_car.radius)
        self.ob3 = np.array([ob3_c[0], ob3_c[1], ob3_d], dtype=np.float32)

        # 観測情報を構築する
        #ob_return = np.hstack([self.ob1, self.ob2, self.ob3])
        ob_return = [self.ob1, self.ob2, self.ob3]

        return ob_return
    
    def step_calc_reward(self, collision_count, eb_flag):

        reward_map = { 'enagy_asumption': -1.0 }

        reward_map['collision'] = -collision_count
#                    if self.training_step == 3:
#                        reward_map['collision'] += -1.0
#                    else:
#                        reward_map['collision'] = -1.0

        # エネルギー体へ接近出来たら報酬
        reward_map['distance_diff'] = 0.0
#        if self.old_ob3 is not None:
#            if self.ob3[2] > 0.5:
#                if abs(ob3_angle) > math.pi/180.0*10.0:
#                    if (self.old_ob3[2] * 0.99) > self.ob3[2]:
#                        reward_map['distance_diff'] = -1.0        #print('angle:', ob3_angle/math.pi*180.0)
#            else:
#                if (self.old_ob3[2] * 1.01) < self.ob3[2]:
#                    # 前の位置よりも近づいているようなら、+1の報酬
#                    reward_map['distance_diff'] = 1.0
#                elif (self.old_ob3[2] * 0.99) > self.ob3[2]:
#                    reward_map['distance_diff'] = -1.0        #print('angle:', ob3_angle/math.pi*180.0)

        # 十分に近づいたら、距離の情報は報酬にしない。EBを入手するには向きが重要。
        reward_map['in_angle'] = 0.0
        ob3_angle = math.atan2( self.ob3[1], self.ob3[0])
        if self.ob3[2] > 0.5:
            reward_map['distance'] = 0.0
            if abs(ob3_angle) > math.pi/180.0*60.0:
                reward_map['in_angle'] = -1.0
            elif abs(ob3_angle) < math.pi/180.0*10.0:
                reward_map['in_angle'] = 1.0    # 報酬体系にメリハリをつけて、価値の判断がはっきりできるようにする。
        else:
            reward_map['distance'] = self.ob3[2] - 0.5
            if abs(ob3_angle) < math.pi/180.0*10.0:
                reward_map['in_angle'] = 1.0    # 報酬体系にメリハリをつけて、価値の判断がはっきりできるようにする。

        if self.old_dist is not None:
#            if (self.old_dist * 1.10) < self.ob3[2]:
            if self.old_dist < self.ob3[2]:
                # 前の位置よりも近づいているようなら、+1の報酬
                reward_map['distance_diff'] = 1.0
                self.old_dist = self.ob3[2]
            else:
                # 前の位置よりも近づいていないなら、角度評価のプラス報酬はつけない
                if reward_map['in_angle'] > 0.0:
                   reward_map['in_angle'] = 0.0 
                if (self.old_dist * 0.98) > self.ob3[2]:
                    reward_map['distance_diff'] = -1.0        #print('angle:', ob3_angle/math.pi*180.0)
                    self.old_dist = self.ob3[2] / 0.98
        else:
            self.old_dist = self.ob3[2]
#        if self.old_ob3 is not None:
#            if (self.old_ob3[2] * 1.01) < self.ob3[2]:
#                # 前の位置よりも近づいているようなら、+1の報酬
#                reward_map['distance_diff'] = 1.0
#            elif (self.old_ob3[2] * 0.99) > self.ob3[2]:
#                reward_map['distance_diff'] = -1.0        #print('angle:', ob3_angle/math.pi*180.0)

        # 報酬を求める
        if eb_flag:  # ご飯にありつけた
            self.twn.enagy += TWN_BoxGardenEnv.eb_enagy
            reward_map['enagy_up'] = TWN_BoxGardenEnv.eb_enagy
            if self.twn.enagy > twn.TwoWheelMover.max_enagy:
                # 満腹以上分は、報酬を与えない。
                # reward_map['enagy_up'] = 1.0 - (self.twn.enagy - twn.TwoWheelMover.max_enagy) / TWN_BoxGardenEnv.eb_enagy
                reward_map['enagy_up'] -= self.twn.enagy - twn.TwoWheelMover.max_enagy
                self.twn.enagy = twn.TwoWheelMover.max_enagy
            reward_map['enagy_up'] /= twn.TwoWheelMover.max_enagy
            reward_map['enagy_up'] *= 10.0
            reward_map['enagy_up'] += 1.0

        if self.twn.enagy <= 0.0:
            reward_map['enagy_zero'] = -10.0

        return reward_map
    
    def step_calc_reward2(self, collision_count, eb_flag):

        reward_map = {}

        reward_map['collision'] = -collision_count
#                    if self.training_step == 3:
#                        reward_map['collision'] += -1.0
#                    else:
#                        reward_map['collision'] = -1.0

        # エネルギー体へ接近出来たら報酬
        reward_map['distance'] = self.ob3[2]

        reward_map['in_angle'] = 0.0
        ob3_angle = math.atan2( self.ob3[1], self.ob3[0])
        if abs(ob3_angle) > math.pi/180.0*60.0:
            reward_map['in_angle'] = -1.0
        elif abs(ob3_angle) < math.pi/180.0*10.0:
            reward_map['in_angle'] = 1.0    # 報酬体系にメリハリをつけて、価値の判断がはっきりできるようにする。


        # 報酬を求める
        if eb_flag:  # ご飯にありつけた
            self.twn.enagy += TWN_BoxGardenEnv.eb_enagy
            reward_map['enagy_up'] = TWN_BoxGardenEnv.eb_enagy
            if self.twn.enagy > twn.TwoWheelMover.max_enagy:
                # 満腹以上分は、報酬を与えない。
                # reward_map['enagy_up'] = 1.0 - (self.twn.enagy - twn.TwoWheelMover.max_enagy) / TWN_BoxGardenEnv.eb_enagy
                reward_map['enagy_up'] -= self.twn.enagy - twn.TwoWheelMover.max_enagy
                self.twn.enagy = twn.TwoWheelMover.max_enagy
            reward_map['enagy_up'] /= twn.TwoWheelMover.max_enagy
            reward_map['enagy_up'] *= 10.0
            reward_map['enagy_up'] += 1.0

        if self.twn.enagy <= 0.0:
            reward_map['enagy_zero'] = -10.0

        return reward_map
    
    #def _step(self, action):
    def step(self, action):
        done = False
        pre_enagy = copy.copy(self.twn.enagy)
        self.old_ob1 = self.ob1
        self.old_ob2 = self.ob2
        self.old_ob3 = self.ob3
        
        collision_count, eb_in_flag = self.step_next_state(action)
        
        # 観測情報を集める
        ob_return = self.gather_observation()
        
        # 報酬情報の収集
        reward_map = self.step_calc_reward(collision_count, eb_in_flag)
 #       reward_map = self.step_calc_reward2(collision_count, eb_in_flag)

        # 終了判定
        if eb_in_flag:  # ご飯にありつけた
            done = True         # エピソードを終了する
            self.get_eb_count += 1
            self.logger.critical("Get EB!!!")

        # TWNのエネルギーがなくなったら、終了
        if self.twn.enagy <= 0.0:
            self.logger.critical("No enagy of TWN !!!!!!!!!!!")
            done = True

        self.obs_hist.append(np.hstack(ob_return))
        del self.obs_hist[0]
        
        r_obs = np.hstack(self.obs_hist)
        
        reward = sum(reward_map.values())*self.reward_scalor
#        reward = sum(reward_map.values())
        #self.logger.info( 'reward: {}'.format(reward_map) )
        #print( 'reward: {}'.format(reward_map) )
        
        if done:
            self.current_trainer.check_end_status(ob_return, reward_map, eb_in_flag)

        self.cumulative_value_of_reward = reward + self.cumulative_value_of_reward * 0.99
#        self.cumulative_value_of_reward += reward
        return r_obs, reward, done, {'reward_map': reward_map, 'cumulative_reward': self.cumulative_value_of_reward }
    
    #def _reset(self):
    def reset(self):
        self.twn.Reset()
        self.reset_eb_pos()
        self.ob1 = None
        self.ob2 = None
        self.ob3 = None
        self.old_ob1 = None
        self.old_ob2 = None
        self.old_ob3 = None
        self.old_dist = None
        self.cumulative_value_of_reward = 0.0
        #self.get_eb_count = 0
        self.collision_count = 0

        self.reset_render()
        
        ob_return = self.gather_observation()
        
        self.obs_hist = []
        for i in range(self.obs_hist_num):
            self.obs_hist.append(np.hstack(ob_return))
        
        r_obs = np.hstack(self.obs_hist)
 
        return r_obs

    def reset_render(self, mode='human', close=False):
        if self.twn_bg_draw is not None:
            self.my_car.restart_pos(pos=self.twn.pos, rot_theata=self.twn.GetRotTheata())

    #def _render(self, mode='human', close=False):
    def render(self, mode='human', close=False, add_info=None):
        if self.twn_bg_draw is not None:
            # print('twn rot degree:', self.twn.GetRotTheata()/math.pi*180.0)
            if self.ob1 is None:
                self.my_car.update_pos_attrs(pos=self.twn.pos, rot_theata=self.twn.GetRotTheata(), attrs=[])
            else:
                self.my_car.update_pos_attrs(pos=self.twn.pos, rot_theata=self.twn.GetRotTheata(), attrs=[self.ob1, self.ob2, self.ob3, self.cumulative_value_of_reward, add_info])

    #def _seed(self, seed=1):
    def seed(self, seed=1):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def finish(self):
        self.bg_draw_boot.finish()
        
    def collision_circle2circle(self, c1, r1, c2, r2):
        ans = False
        dif_len = np.linalg.norm(c1 - c2)
        if dif_len < (r1 + r2 + 0.2):
            ans = True
        return ans
    
    def collision_circle2wall(self, c1, r1, w2, r2):
        ans = False
        dif_len = np.linalg.norm(c1 - w2)
        if r2 < (dif_len + r1 + 0.2):
            ans = True
        return ans


    def reset_eb_pos(self):
        self.try_count += 1
        self.logger.info('EB count: {}'.format(self.get_eb_count))
        if self.get_eb_count < 100:
            self.training_step = 0
            self.current_trainer = self.trainers[0]
        elif self.get_eb_count < 200:
            self.training_step = 1
            self.current_trainer = self.trainers[self.try_count % 2]
        elif self.get_eb_count < 400:
            self.training_step = 2
            self.current_trainer = self.trainers[self.try_count % 3]
        else:
            self.training_step = 3
            self.current_trainer = self.trainers[self.try_count % 4]

        self.twn.Reset()
        for collision_retry in range(10):
            new_pos = self.current_trainer.get_new_eb_pos()
            if not self.collision_eb(new_pos):
                #print('step{} EB count: {}, pos {}'.format(self.training_step, self.get_eb_count, new_pos.reshape(1,2)))
                break
        if self.twn_bg_draw is not None:
            self.eb.update_pos_attrs(new_pos, self.eb.radius)

class TWN_BoxGardenEnv1(TWN_BoxGardenEnv):
    '''
    TWN_BoxGardenEnvの特殊版
    '''
    def __init__(self, discrate_action=True):
        super().__init__(discrate_action, history_size=1, renderable=True)
        print('TWN_BoxGardenEnv2: history size {}  renderable=True'.format(self.obs_hist_num))

#    def step(self, action):
#        observation, reward, done, info = super().step(action)
#        r_obs = np.hstack(self.obs_hist)
# 
#        return r_obs, reward, done, info
#
#    def reset(self):
#        super().reset()
#        r_obs = np.hstack(self.obs_hist)
#
#        return r_obs
    
class TWN_BoxGardenEnv2(TWN_BoxGardenEnv):
    '''
    TWN_BoxGardenEnvの特殊版
    '''
    def __init__(self, discrate_action=True):
        super().__init__(discrate_action, history_size=1, renderable=False)
        print('TWN_BoxGardenEnv2: history size {}  renderable=False'.format(self.obs_hist_num))

#    def step(self, action):
#        observation, reward, done, info = super().step(action)
#        r_obs = np.hstack(self.obs_hist)
# 
#        return r_obs, reward, done, info
#
#    def reset(self):
#        super().reset()
#        r_obs = np.hstack(self.obs_hist)
#
#        return r_obs
    
class TWN_BoxGardenEnv3(TWN_BoxGardenEnv):
    '''
    TWN_BoxGardenEnvの特殊版
    観測情報の過去情報も得られる版
    '''
    def __init__(self, discrate_action=True):
        super().__init__(discrate_action, history_size=10)
        print("TWN_BoxGardenEnv3: history size ", self.obs_hist_num)

#    def step(self, action):
#        observation, reward, done, info = super().step(action)
#        r_obs = np.hstack(self.obs_hist)
# 
#        return r_obs, reward, done, info
#
#    def reset(self):
#        super().reset()
#        r_obs = np.hstack(self.obs_hist)
#
#        return r_obs
    

class TWN_BoxGardenEnv4(TWN_BoxGardenEnv):
    '''
    TWN_BoxGardenEnvの特殊版
    Action spaceが連続値空間版
    '''
    def __init__(self):
        super().__init__(discrate_action=False, history_size=1, renderable=True)
        print("TWN_BoxGardenEnv4: Continuous action space: history size ", self.obs_hist_num)

class TWN_BoxGardenEnv5(TWN_BoxGardenEnv):
    '''
    TWN_BoxGardenEnvの特殊版
    Action spaceが連続値空間版
    '''
    def __init__(self):
        super().__init__(discrate_action=False, history_size=1, renderable=False)
        print("TWN_BoxGardenEnv5: Continuous action space: history size ", self.obs_hist_num)

if __name__ == '__main__':
    mp.freeze_support()

    logging.basicConfig(format='%(asctime)s/%(module)s/%(name)s/%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
#    drawobj_base.debug_logger.setLevel(logging.CRITICAL)
    
    twn_bgenv = TWN_BoxGardenEnv()
    
    twn_bgenv._render()
    for i in range(100):
        o, r, d, info = twn_bgenv._step(np.array([0.5, 0.3]))
        #print("o")
        #print(o)
        #print("r")
        #print(r)
        #print("d")
        #print(d)
        print("i: ",i)
        twn_bgenv._render()
    
    twn_bgenv._reset()
    twn_bgenv._render()
    for i in range(1000):
        o, r, d, info = twn_bgenv._step(np.array([0.5, 0.5]))
        #print("o")
        #print(o)
        #print("r")
        #print(r)
        #print("d")
        #print(d)
        print("i: ",i)
        twn_bgenv._render()
    
    twn_bgenv.finish()
