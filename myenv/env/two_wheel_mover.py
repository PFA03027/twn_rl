# -*- coding: utf-8 -*-
"""
TwoWheelMover
Created on Sun Jul 15 17:42:04 2018

@author: alpha

"""

import math
import numpy as np
from numpy import sin, cos

class TwoWheelMover():
    max_wheel_vec =  1.0
    min_wheel_vec = -1.0
    max_wheel_acc =  0.3
    min_wheel_acc = -0.3
    max_enagy = 3000
    delta_t = 1.0
    def __init__(self):
        self.pos = np.zeros((2), dtype=np.float32).reshape(2,1)
        self.direction = np.array([0.0, 1.0], dtype=np.float32).reshape(2,1)
        self.wheel_vec = np.zeros((2), dtype=np.float32)
        self.wheel_acc = np.zeros((2), dtype=np.float32)
        self.wheel_length = 1.0
        self.enagy = TwoWheelMover.max_enagy
        self.velocity = np.zeros((2), dtype=np.float32).reshape(2,1)

    def Reset(self):
        self.pos = np.zeros((2), dtype=np.float32).reshape(2,1)
        self.direction = np.array([0.0, 1.0], dtype=np.float32).reshape(2,1)
        self.wheel_vec = np.zeros((2), dtype=np.float32)
        self.wheel_acc = np.zeros((2), dtype=np.float32)
        self.enagy = TwoWheelMover.max_enagy
        self.velocity = np.zeros((2), dtype=np.float32).reshape(2,1)

    def Reset2(self):
        self.enagy = TwoWheelMover.max_enagy

    def SetWheelVec(self, w1, w2):
        self.wheel_vec = np.array((w1, w2), dtype=np.float32)
        self.wheel_vec = np.clip( self.wheel_vec, TwoWheelMover.min_wheel_vec, TwoWheelMover.max_wheel_vec )

    def SetWheelAccel(self, w1_a, w2_a):
        self.wheel_acc = np.array((w1_a, w2_a), dtype=np.float32)
        self.wheel_acc = np.clip( self.wheel_acc, TwoWheelMover.min_wheel_acc, TwoWheelMover.max_wheel_acc )

    def SetAccWheel(self, accel, wheel):
        # いわゆる車のアクセル/ブレーキ/ハンドルでの操作
        self.wheel_vec = np.array((accel, accel), dtype=np.float32)
        if wheel > 0.05:        # 右へ曲がる
            self.wheel_vec[1] -= self.wheel_vec[1] * wheel
        elif wheel < -0.05:        # 左へ曲がる
            self.wheel_vec[0] -= self.wheel_vec[0] * (-wheel)
        self.wheel_vec = np.clip( self.wheel_vec, TwoWheelMover.min_wheel_vec, TwoWheelMover.max_wheel_vec )

    def SetRotOnly(self, left_rot, right_rot):
        # 右か、左に回転するだけの特殊操作用のモデル。
        f_w_v = left_rot - right_rot
        self.SetWheelVec(-f_w_v, f_w_v)

    def _calcRotateSpeed(self):
        if math.fabs(self.wheel_vec[0] - self.wheel_vec[1]) < 0.00001:
            rot_speed = 0.0
            rc = None
        else:
            rc = (self.wheel_vec[0] + self.wheel_vec[1]) / (self.wheel_vec[0] - self.wheel_vec[1]) * self.wheel_length
            rot_speed = -(self.wheel_vec[0] - self.wheel_vec[1]) / (2.0 * self.wheel_length)
            
        return rot_speed, rc

    def GetRotateSpeed(self):
        ans, rc = self._calcRotateSpeed()
        return ans

    def GetRotateSpeedMinMax(self):
        ans = (TwoWheelMover.max_wheel_vec - TwoWheelMover.min_wheel_vec) / (2.0 * self.wheel_length)
        return [-ans, ans]
    
    def GetRotTheata(self):
        return math.atan2(self.direction[1], self.direction[0])

    def MoveNext(self):
        "Move to new position as new turn"
        self.enagy -= TwoWheelMover.delta_t + np.sum(np.abs(self.wheel_vec))*0.2
        if self.enagy < 0:
            self.enagy = 0
        self.wheel_vec += self.wheel_acc
        self.wheel_vec = np.clip( self.wheel_vec, TwoWheelMover.min_wheel_vec, TwoWheelMover.max_wheel_vec )
        rot_speed, rc = self._calcRotateSpeed()
        #print( "rot_speed:", rot_speed, "  rc:", rc )
        if rc is None:
            self.pos = self.pos + self.direction * (self.wheel_vec[0] * TwoWheelMover.delta_t)
            self.velocity = self.direction.copy() * self.wheel_vec[0].copy()
        else:
            rc_direct = self.direction.copy()
            #print( "rc_direct:", rc_direct )
            rc_direct = np.matrix(((0.0, 1.0), (-1.0, 0.0))).dot(rc_direct)
            #print( "rc_direct:", rc_direct )
            rc_direct = rc_direct * rc
            #print( "rc_direct:", rc_direct )
            rc_pos = self.pos + rc_direct
            #print( "pos:", self.pos, "rc_pos:", rc_pos )
            rc_direct = rc_direct * -1.0
            #print( "rc_direct:", rc_direct )
            theata =  rot_speed * TwoWheelMover.delta_t
            #print( "theata[rad]:", theata, "theata[dig]:", theata/np.pi*180.0 )
            rot = np.matrix(((cos(theata), -sin(theata)), (sin(theata), cos(theata))))            #2次元回転
            #print( "rot:", rot )
            rotated_rc_direct = rot.dot(rc_direct)
            #print( "rotated_rc_direct:", rotated_rc_direct )
            self.pos = rc_pos + rotated_rc_direct
            self.direction = rot.dot(self.direction)
            self.direction = self.direction / np.linalg.norm(self.direction)
            
            v_vec = self.direction.copy()
            self.velocity = v_vec * (-(rc * rot_speed))

class TwoWheelMover_SetWheelVec(TwoWheelMover):
    def SetOperationParam(self, w1, w2):
        super().SetWheelVec(w1, w2)
    
class TwoWheelMover_SetWheelAccel(TwoWheelMover):
    def SetOperationParam(self, w1, w2):
        super().SetWheelAccel(w1, w2)
    
class TwoWheelMover_SetRotOnly(TwoWheelMover):
    def SetOperationParam(self, w1, w2):
        super().SetRotOnly(w1, w2)
    


if __name__ == '__main__':
    target_pos = np.array([0.0, 100.0])
    twm = TwoWheelMover()
    print( "pos:", twm.pos.reshape(1,2), "direction:", twm.direction.reshape(1,2), "wheel vec:", twm.wheel_vec.reshape(1,2), "Rot speed: ", twm.GetRotateSpeed(), "Rot: ", twm.GetRotTheata()/math.pi*180.0 )

    print( "b pos:", twm.pos.reshape(1,2), "direction:", twm.direction.reshape(1,2), "wheel vec:", twm.wheel_vec.reshape(1,2), "Rot speed: ", twm.GetRotateSpeed(), "Rot: ", twm.GetRotTheata()/math.pi*180.0 )
    twm.SetWheelVec(1.0, 1.0)
    print( "s pos:", twm.pos.reshape(1,2), "direction:", twm.direction.reshape(1,2), "wheel vec:", twm.wheel_vec.reshape(1,2), "Rot speed: ", twm.GetRotateSpeed(), "Rot: ", twm.GetRotTheata()/math.pi*180.0 )
    twm.MoveNext()
    print( "a pos:", twm.pos.reshape(1,2), "direction:", twm.direction.reshape(1,2), "wheel vec:", twm.wheel_vec.reshape(1,2), "Rot speed: ", twm.GetRotateSpeed(), "Rot: ", twm.GetRotTheata()/math.pi*180.0 )

    print( "b pos:", twm.pos.reshape(1,2), "direction:", twm.direction.reshape(1,2), "wheel vec:", twm.wheel_vec.reshape(1,2), "Rot speed: ", twm.GetRotateSpeed(), "Rot: ", twm.GetRotTheata()/math.pi*180.0 )
    twm.SetWheelVec(-1.0, -1.0)
    print( "s pos:", twm.pos.reshape(1,2), "direction:", twm.direction.reshape(1,2), "wheel vec:", twm.wheel_vec.reshape(1,2), "Rot speed: ", twm.GetRotateSpeed(), "Rot: ", twm.GetRotTheata()/math.pi*180.0 )
    twm.MoveNext()
    print( "a pos:", twm.pos.reshape(1,2), "direction:", twm.direction.reshape(1,2), "wheel vec:", twm.wheel_vec.reshape(1,2), "Rot speed: ", twm.GetRotateSpeed(), "Rot: ", twm.GetRotTheata()/math.pi*180.0 )

    print( "b pos:", twm.pos.reshape(1,2), "direction:", twm.direction.reshape(1,2), "wheel vec:", twm.wheel_vec.reshape(1,2), "Rot speed: ", twm.GetRotateSpeed(), "Rot: ", twm.GetRotTheata()/math.pi*180.0 )
    twm.SetWheelVec(-1.0, 0.0)
    print( "s pos:", twm.pos.reshape(1,2), "direction:", twm.direction.reshape(1,2), "wheel vec:", twm.wheel_vec.reshape(1,2), "Rot speed: ", twm.GetRotateSpeed(), "Rot: ", twm.GetRotTheata()/math.pi*180.0 )
    twm.MoveNext()
    print( "a pos:", twm.pos.reshape(1,2), "direction:", twm.direction.reshape(1,2), "wheel vec:", twm.wheel_vec.reshape(1,2), "Rot speed: ", twm.GetRotateSpeed(), "Rot: ", twm.GetRotTheata()/math.pi*180.0 )

    print( "b pos:", twm.pos.reshape(1,2), "direction:", twm.direction.reshape(1,2), "wheel vec:", twm.wheel_vec.reshape(1,2), "Rot speed: ", twm.GetRotateSpeed(), "Rot: ", twm.GetRotTheata()/math.pi*180.0 )
    twm.SetWheelVec(0.0, -1.0)
    print( "s pos:", twm.pos.reshape(1,2), "direction:", twm.direction.reshape(1,2), "wheel vec:", twm.wheel_vec.reshape(1,2), "Rot speed: ", twm.GetRotateSpeed(), "Rot: ", twm.GetRotTheata()/math.pi*180.0 )
    twm.MoveNext()
    print( "a pos:", twm.pos.reshape(1,2), "direction:", twm.direction.reshape(1,2), "wheel vec:", twm.wheel_vec.reshape(1,2), "Rot speed: ", twm.GetRotateSpeed(), "Rot: ", twm.GetRotTheata()/math.pi*180.0 )

    print( "b pos:", twm.pos.reshape(1,2), "direction:", twm.direction.reshape(1,2), "wheel vec:", twm.wheel_vec.reshape(1,2), "Rot speed: ", twm.GetRotateSpeed(), "Rot: ", twm.GetRotTheata()/math.pi*180.0 )
    twm.SetWheelVec(1.0, 0.0)
    print( "s pos:", twm.pos.reshape(1,2), "direction:", twm.direction.reshape(1,2), "wheel vec:", twm.wheel_vec.reshape(1,2), "Rot speed: ", twm.GetRotateSpeed(), "Rot: ", twm.GetRotTheata()/math.pi*180.0 )
    twm.MoveNext()
    print( "a pos:", twm.pos.reshape(1,2), "direction:", twm.direction.reshape(1,2), "wheel vec:", twm.wheel_vec.reshape(1,2), "Rot speed: ", twm.GetRotateSpeed(), "Rot: ", twm.GetRotTheata()/math.pi*180.0 )

    print( "b pos:", twm.pos.reshape(1,2), "direction:", twm.direction.reshape(1,2), "wheel vec:", twm.wheel_vec.reshape(1,2), "Rot speed: ", twm.GetRotateSpeed(), "Rot: ", twm.GetRotTheata()/math.pi*180.0 )
    twm.SetWheelVec(0.0, 1.0)
    print( "s pos:", twm.pos.reshape(1,2), "direction:", twm.direction.reshape(1,2), "wheel vec:", twm.wheel_vec.reshape(1,2), "Rot speed: ", twm.GetRotateSpeed(), "Rot: ", twm.GetRotTheata()/math.pi*180.0 )
    twm.MoveNext()
    print( "a pos:", twm.pos.reshape(1,2), "direction:", twm.direction.reshape(1,2), "wheel vec:", twm.wheel_vec.reshape(1,2), "Rot speed: ", twm.GetRotateSpeed(), "Rot: ", twm.GetRotTheata()/math.pi*180.0 )

    print( "b pos:", twm.pos.reshape(1,2), "direction:", twm.direction.reshape(1,2), "wheel vec:", twm.wheel_vec.reshape(1,2), "Rot speed: ", twm.GetRotateSpeed(), "Rot: ", twm.GetRotTheata()/math.pi*180.0 )
    twm.SetWheelVec(1.0, 0.5)
    print( "s pos:", twm.pos.reshape(1,2), "direction:", twm.direction.reshape(1,2), "wheel vec:", twm.wheel_vec.reshape(1,2), "Rot speed: ", twm.GetRotateSpeed(), "Rot: ", twm.GetRotTheata()/math.pi*180.0 )
    twm.MoveNext()
    print( "a pos:", twm.pos.reshape(1,2), "direction:", twm.direction.reshape(1,2), "wheel vec:", twm.wheel_vec.reshape(1,2), "Rot speed: ", twm.GetRotateSpeed(), "Rot: ", twm.GetRotTheata()/math.pi*180.0 )

    twm.MoveNext()
    print( "a pos:", twm.pos.reshape(1,2), "direction:", twm.direction.reshape(1,2), "wheel vec:", twm.wheel_vec.reshape(1,2), "Rot speed: ", twm.GetRotateSpeed(), "Rot: ", twm.GetRotTheata()/math.pi*180.0 )
    twm.MoveNext()
    print( "a pos:", twm.pos.reshape(1,2), "direction:", twm.direction.reshape(1,2), "wheel vec:", twm.wheel_vec.reshape(1,2), "Rot speed: ", twm.GetRotateSpeed(), "Rot: ", twm.GetRotTheata()/math.pi*180.0 )
    twm.MoveNext()
    print( "a pos:", twm.pos.reshape(1,2), "direction:", twm.direction.reshape(1,2), "wheel vec:", twm.wheel_vec.reshape(1,2), "Rot speed: ", twm.GetRotateSpeed(), "Rot: ", twm.GetRotTheata()/math.pi*180.0 )

    twm.Reset()
    print( "a pos:", twm.pos.reshape(1,2), "direction:", twm.direction.reshape(1,2), "wheel vec:", twm.wheel_vec.reshape(1,2), "Rot speed: ", twm.GetRotateSpeed(), "Rot: ", twm.GetRotTheata()/math.pi*180.0 )

    len_vec   = target_pos.reshape(-1) - twm.pos.reshape(-1)
    ans_len   = np.linalg.norm( len_vec )
    print( "pos: ", twm.pos.reshape(-1), "direction:", twm.direction.reshape(-1), "len vec:", len_vec.reshape(-1) )
    cr_value  = np.cross(twm.direction.reshape(-1),len_vec.reshape(-1))
    dot_value = np.inner(twm.direction.reshape(-1),len_vec.reshape(-1))
    ans_angle = math.atan2(cr_value, dot_value)
    print( "pos: ", twm.pos.reshape(1,2), "direction:", twm.direction.reshape(1,2), "len vec:", len_vec.reshape(1,2), "angle: ", ans_angle/3.14*180.0 )

 
    twm2 = TwoWheelMover()
    twm2.SetWheelAccel(0.1, 0.09)
    for i in range(200):
        twm2.MoveNext()
        print( "pos:", twm2.pos.reshape(1,2), "direction:", twm2.direction.reshape(1,2), "wheel vec:", twm2.wheel_vec.reshape(1,2), "Rot: ", twm2.GetRotateSpeed(), "Enagy: ", twm2.enagy )

