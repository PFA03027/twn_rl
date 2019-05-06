# -*- coding: utf-8 -*-
"""
箱庭内に配置される物のライブラリ
Created on Sun Jul 15 17:42:04 2018

@author: alpha

基本コンセプト
視線の始点と視線ベクトルとの交差を計算し、始点から物までの距離を算出する機能を有する

"""

import math
import numpy as np
from enum import Enum

import time

from . import BoxGarden
from . import BoxGarden_draw
from . import drawing_trace2 as dt2

class circle_object(BoxGarden.bg_object):
    '''
    円型の箱庭オブジェクト
    '''
    def __init__(self, pos=np.zeros(2, dtype=np.float32), r=1.0):
        '''
        第1引数  2つの要素を持つnumpyの配列。円の中心の座標
        第2引数  半径
        '''
        super().__init__(pos)
        self.radius = r

        self.target_bg_draw = None
        

    def culc_distance_to_cross_Nray(self, start_point):
        '''
        第1引数  2つの要素を持つnumpyの配列。交差計算の始点。
        
        戻り値   ３つの要素を持つリスト
               第1要素   bg_obj_RC型。交差しているかどうかの判定結果のリスト
               第2要素   始点からの距離の1Dのnp配列。ただし、交差していない場合(bg_obj_RC.CROSSED以外)は意味を持たない
               第3要素   2つの要素を持つnumpyの配列。交差計算で使用された媒介変数。[2,n]の形
                        第1行n列 n番目のrayにおける円の中心からの距離に相当する値となる媒介変数。ベクトル演算を行っているため、負の値を取り得る。純粋に距離が必要の場合は、絶対値を取る必要がある
                        第2行n列 n番目のrayにおける始点から、円の中心からの垂線の交差点までの距離に相当する値となる媒介変数。ベクトル演算を行っているため、負の値を取り得る。純粋に距離が必要の場合は、絶対値を取る必要がある
        補足    基本クラスは、点を表す物体となるため、交差しない。
        '''
        local_sp_diff = self.pos.reshape(2,1) - start_point.reshape(2,1)
        pv_st = self.inv_mat_ray.dot(local_sp_diff).reshape(self.vls_size[1],2).T
        
        y = pv_st[1,:]
        y = self.radius**2 - np.multiply(y,y)
        y[y < 0.0] = 0.0
        x = np.abs(pv_st[0,:]) - np.sqrt(y)
        x = x.reshape(1, self.vls_size[1])
            
        ret_type = []
        for idx in range(self.vls_size[1]):
            if(abs(pv_st[1,idx]) > self.radius):
                ret_type.append(BoxGarden.bg_obj_RC.NOT_CROSSED)   # 始点からの視線ベクトルは、円に交差しない
            else:
                if(x[0,idx] < 0.0):
                    ret_type.append(BoxGarden.bg_obj_RC.IN_OBJECT)   # 始点は、円の中
                    #print('ID: ', self.id, 'PID: ', self.pid, ' <= circle_object,  ret: ', ret_type, 'pv_st: ', pv_st)
                elif(pv_st[0,idx] > 0.0):
                    ret_type.append(BoxGarden.bg_obj_RC.CROSSED)
                else:
                    ret_type.append(BoxGarden.bg_obj_RC.NOT_CROSSED)

        return ret_type, x, pv_st

    '''
    円柱の壁を表す箱庭オブジェクトの描画用オブジェクト(クラス内クラス定義)
    '''
    class circle_object_draw(BoxGarden_draw.bg_object_draw):
        def __init__(self, pos=np.zeros(2, dtype=np.float32), r=1.0, ptidx=0):
            '''
            第1引数  2つの要素を持つnumpyの配列。円の中心の座標
            第2引数  半径
            '''
            super().__init__()
            self.pos = pos
            self.radius = r
            self.plot_target_idx = ptidx
        
        def process_add(self, msg, bg_draw):
            ax = bg_draw.scn.subplot_list[self.plot_target_idx].ax
            
            self.my_do = dt2.drawobj_circle(ax, self.pos[0], self.pos[1], self.radius, color='b')
            bg_draw.scn.subplot_list[self.plot_target_idx].append_drawobj(self.my_do)
    
        def process_update(self, msg, bg_draw):
            self.my_do.update(msg.updated_attrs['pos'][0,0], msg.updated_attrs['pos'][1,0], msg.updated_attrs['radius'])

        def update_subplot(self, data):
            pass
    
    def send_drawobj(self, bg_draw):
        '''
        描画用オブジェクトをBoxGarden_drawに送る
        '''
        self.target_bg_draw = bg_draw

        obj = circle_object.circle_object_draw(self.pos, self.radius)

        self.do_id = obj.id     # 相手先のidを保持しておく
        mes = BoxGarden_draw.bg_message_base()
        mes.command_ADD(obj)
        self.target_bg_draw.send_msg(mes)

    def update_pos_attrs(self, pos, r):
        self.pos = pos.reshape(2,1).astype(np.float32)
        self.radius = r
        mes = BoxGarden_draw.bg_message_base()
        mes.command_UPDATE(self.do_id, {'pos': self.pos, 'radius': self.radius})
        self.target_bg_draw.send_msg(mes)

class circle_wall(BoxGarden.bg_object):
    '''
    円型の壁を表す箱庭オブジェクト
    箱庭の外壁を表すため、交差計算では、円の内側に視点が存在する必要がある。
    円の外側に始点が存在する場合、bg_obj_RC.IN_OBJECTと判定する。
    '''
    def __init__(self, pos=np.zeros(2, dtype=np.float32), r=100.0):
        '''
        第1引数  2つの要素を持つnumpyの配列。円の中心の座標
        第2引数  半径
        '''
        super().__init__(pos)
        self.radius = r
        
        self.target_bg_draw = None
        
    def culc_distance_to_cross_Nray(self, start_point):
        '''
        第1引数  2つの要素を持つnumpyの配列。交差計算の始点。
        
        戻り値   ３つの要素を持つリスト
               第1要素   bg_obj_RC型。交差しているかどうかの判定結果のリスト
               第2要素   始点からの距離の1Dのnp配列[1,n]。ただし、交差していない場合(bg_obj_RC.CROSSED以外)は意味を持たない
               第3要素   2つの要素を持つnumpyの配列。交差計算で使用された媒介変数。[2,n]の形
                        第1行n列 n番目のrayにおける円の中心からの距離に相当する値となる媒介変数。ベクトル演算を行っているため、負の値を取り得る。純粋に距離が必要の場合は、絶対値を取る必要がある
                        第2行n列 n番目のrayにおける始点から、円の中心からの垂線の交差点までの距離に相当する値となる媒介変数。ベクトル演算を行っているため、負の値を取り得る。純粋に距離が必要の場合は、絶対値を取る必要がある
        補足    基本クラスは、点を表す物体となるため、交差しない。
        '''
        local_sp_diff = self.pos.reshape(2,1) - start_point.reshape(2,1)
        pv_st = self.inv_mat_ray.dot(local_sp_diff).reshape(self.vls_size[1],2).T

        y = pv_st[1,:].copy()
        y = self.radius**2 - np.multiply(y, y)
        y[y < 0.0] =  0.0
        y = np.sqrt(y)
        x = y + pv_st[0,:].copy()
        x = x.reshape(1, self.vls_size[1])
        y = y.reshape(1, self.vls_size[1])

        ret_type = []
        for idx in range(self.vls_size[1]):
            if(abs(pv_st[1,idx]) > self.radius):
                ret_type.append(BoxGarden.bg_obj_RC.IN_OBJECT)
            else:
                if(pv_st[0,idx] >= 0.0):
                    if y[0,idx] >= pv_st[0,idx]:
                        ret_type.append(BoxGarden.bg_obj_RC.CROSSED)
                    else:
                        ret_type.append(BoxGarden.bg_obj_RC.IN_OBJECT)   # 始点は、円の外側
                        #print('ID: ', self.id, 'PID: ', self.pid, ' <= circle_wall,  ret: ', ret_type, ' 始点は、円の外側1 pv_st: ', pv_st)
                else:
                    if y[0,idx] >= -pv_st[0,idx]:
                        ret_type.append(BoxGarden.bg_obj_RC.CROSSED)
                    else:
                        ret_type.append(BoxGarden.bg_obj_RC.IN_OBJECT)   # 始点は、円の外側
                        #print('ID: ', self.id, 'PID: ', self.pid, ' <= circle_wall,  ret: ', ret_type, ' 始点は、円の外側2 pv_st: ', pv_st)

        #x = x.reshape(self.vls_size[1])

        return ret_type, x, pv_st

    '''
    円型の壁を表す箱庭オブジェクトの描画用オブジェクト(クラス内クラス定義)
    '''
    class circle_wall_draw(BoxGarden_draw.bg_object_draw):
        def __init__(self, pos=np.zeros(2, dtype=np.float32), r=1.0, ptidx=0):
            '''
            第1引数  2つの要素を持つnumpyの配列。円の中心の座標
            第2引数  半径
            '''
            super().__init__()
            self.pos = pos
            self.radius = r
            self.plot_target_idx = ptidx
        
        def process_add(self, msg, bg_draw):
            ax = bg_draw.scn.subplot_list[self.plot_target_idx].ax
            
            self.my_do = dt2.drawobj_circle(ax, self.pos[0], self.pos[1], self.radius, linestyle='--')
            bg_draw.scn.subplot_list[self.plot_target_idx].append_drawobj(self.my_do)

        def process_update(self, msg, bg_draw):
            self.my_do.update(msg.updated_attrs['pos'][0,0], msg.updated_attrs['pos'][1,0], msg.updated_attrs['radius'])
    
        def update_subplot(self, data):
            pass
    
    def send_drawobj(self, bg_draw):
        '''
        描画用オブジェクトをBoxGarden_drawに送る
        '''
        self.target_bg_draw = bg_draw

        obj = circle_wall.circle_wall_draw(self.pos, self.radius)

        self.do_id = obj.id     # 相手先のidを保持しておく
        mes = BoxGarden_draw.bg_message_base()
        mes.command_ADD(obj)
        self.target_bg_draw.send_msg(mes)

    def update_pos_attrs(self, pos, r):
        self.pos = pos.reshape(2,1).astype(np.float32)
        self.radius = r
        mes = BoxGarden_draw.bg_message_base()
        mes.command_UPDATE(self.do_id, {'pos': self.pos, 'radius': self.radius})
        self.target_bg_draw.send_msg(mes)

# 以下は、テスト用コード

class _my_car_obj(circle_object):
    def __init__(self, pos=np.zeros(2, dtype=np.float32), rot_theata=0.0):
        '''
        第1引数  2つの要素を持つnumpyの配列。円の中心の座標
        '''
        super().__init__(pos, r=1.0)
        self.rot_theata = rot_theata

        self.target_bg_draw = None
        
    '''
    自車オブジェクトの描画用オブジェクトへのメッセージID
    '''
    class my_car_msg_ID(Enum):
        '''
        メッセージID
        '''
        INVALID = -1
        RESTART = 1
        UPDATE_POS_ATTRS = 2

    '''
    自車オブジェクトの描画用オブジェクトへのメッセージクラスの定義(クラス内クラス定義)
    '''
    class my_car_msg(BoxGarden_draw.bg_message_base):
        def __init__(self, target_id):
            super().__init__(mid=BoxGarden_draw.bg_msg_ID.COMMON, target_id=target_id)
            self.my_car_cmd = _my_car_obj.my_car_msg_ID.INVALID
        
        def restart_pos(self, pos=np.zeros(2, dtype=np.float32), rot_theata=0.0):
            self.my_car_cmd = _my_car_obj.my_car_msg_ID.RESTART
            self.new_pos = pos
            self.new_rot_theata = rot_theata

        def update_pos_attrs(self, pos, rot_theata, attrs):
            self.my_car_cmd = _my_car_obj.my_car_msg_ID.UPDATE_POS_ATTRS
            self.new_pos = pos
            self.new_rot_theata = rot_theata
            self.attrs = attrs

    '''
    自車オブジェクトを表す箱庭オブジェクトの描画用オブジェクト(クラス内クラス定義)
    '''
    class my_car_draw(BoxGarden_draw.bg_object_draw):
        def __init__(self, pos=np.zeros(2, dtype=np.float32), ptidx=0):
            '''
            第1引数  2つの要素を持つnumpyの配列。円の中心の座標
            第2引数  半径
            '''
            super().__init__()
            self.pos = pos
            self.plot_target_idx = ptidx
        
        def process_add(self, msg, bg_draw):
            ax = bg_draw.scn.subplot_list[self.plot_target_idx].ax
            
            self.my_do = dt2.drawobj_empty()
            self.my_do.move = self.pos
            dt2.drawobj_circle(ax, 0.0, 0.0, 1.0, color='r', linestyle='-', parent=self.my_do)
            dt2.drawobj_triangle(ax, 0.0, 0.0, -0.5, -math.sqrt(0.75), 0.5, -math.sqrt(0.75), parent=self.my_do)
            
            bg_draw.scn.subplot_list[self.plot_target_idx].append_drawobj(self.my_do)


        def process_msg(self, msg, bg_draw):
            if msg.my_car_cmd is not None:
                if msg.my_car_cmd == _my_car_obj.my_car_msg_ID.RESTART:
                    self.my_do.move = msg.new_pos
                    self.my_do.rot_theata = msg.new_rot_theata
                elif msg.my_car_cmd == _my_car_obj.my_car_msg_ID.UPDATE_POS_ATTRS:
                    self.my_do.move = msg.new_pos
                    self.my_do.rot_theata = msg.new_rot_theata
                    self.attrs = msg.attrs


        def update_subplot(self, data):
            pass
    
    def send_drawobj(self, bg_draw):
        '''
        描画用オブジェクトをBoxGarden_drawに送る
        '''
        self.target_bg_draw = bg_draw

        obj = _my_car_obj.my_car_draw(self.pos)

        self.do_id = obj.id     # 相手先のidを保持しておく
        mes = BoxGarden_draw.bg_message_base()
        mes.command_ADD(obj)
        self.target_bg_draw.send_msg(mes)

    def restart_pos(self, pos=np.zeros(2, dtype=np.float32), rot_theata=0.0):
        mes = _my_car_obj.my_car_msg(self.do_id)
        mes.restart_pos(pos,rot_theata)
        self.target_bg_draw.send_msg(mes)

    def update_pos_attrs(self, pos, rot_theata, attrs):
        mes = _my_car_obj.my_car_msg(self.do_id)
        mes.update_pos_attrs(pos,rot_theata,attrs)
        self.target_bg_draw.send_msg(mes)


class _BoxGarden_draw_test(BoxGarden_draw.BoxGarden_draw):
    def __init__(self):
        super().__init__([1,1])

    def start_screen(self):
        super().start_screen()

        self.scn.subplot_list[0].ax.set_xlim(-20, 20)
        self.scn.subplot_list[0].ax.set_ylim(-20, 20)
        self.scn.subplot_list[0].ax.set_xlabel('x')
        self.scn.subplot_list[0].ax.set_ylabel('y')
        self.scn.subplot_list[0].ax.grid()


        
if __name__ == '__main__':
    co1 = circle_object(np.array([1.0,10.0]), 2.0)

    rt, l, pv_ts = co1.culc_distance_to_cross(np.array([0.0,0.0]), np.array([0.0,1.0]))
    print( "l:", l, "return type", rt, "parametric variable [t, s]:", pv_ts.reshape(-1) )

    rt, l, pv_ts = co1.culc_distance_to_cross(np.array([1.0,0.0]), np.array([0.0,1.0]))
    print( "l:", l, "return type", rt, "parametric variable [t, s]:", pv_ts.reshape(-1) )
    
    rt, l, pv_ts = co1.culc_distance_to_cross(np.array([4.0,0.0]), np.array([0.0,1.0]))
    print( "l:", l, "return type", rt, "parametric variable [t, s]:", pv_ts.reshape(-1) )
    
    rt, l, pv_ts = co1.culc_distance_to_cross(np.array([0.0,9.0]), np.array([0.0,1.0]))
    print( "l:", l, "return type", rt, "parametric variable [t, s]:", pv_ts.reshape(-1) )

    co2 = circle_wall(np.array([1.0,10.0]), 12.0)
    rt, l, pv_ts = co2.culc_distance_to_cross(np.array([0.0,0.0]), np.array([0.0,1.0]))
    print( "l:", l, "return type", rt, "parametric variable [t, s]:", pv_ts.reshape(-1) )

    rt, l, pv_ts = co2.culc_distance_to_cross(np.array([10.0,0.0]), np.array([0.0,1.0]))
    print( "l:", l, "return type", rt, "parametric variable [t, s]:", pv_ts.reshape(-1) )
    
    bg = BoxGarden.BoxGarden()
    bg.append_bg_obj(co1)
    bg.append_bg_obj(co2)

    rt, l, pv_ts = bg.culc_nearest_distance_to_cross(np.array([0.0,0.0]), np.array([0.0,1.0]))
    print( "l:", l, "return type", rt, "ID: ", pv_ts )

    rt, l, pv_ts = bg.culc_nearest_distance_to_cross(np.array([1.0,0.0]), np.array([0.0,1.0]))
    print( "l:", l, "return type", rt, "ID: ", pv_ts )
    
    rt, l, pv_ts = bg.culc_nearest_distance_to_cross(np.array([4.0,0.0]), np.array([0.0,1.0]))
    print( "l:", l, "return type", rt, "ID: ", pv_ts )
    
    rt, l, pv_ts = bg.culc_nearest_distance_to_cross(np.array([0.0,9.0]), np.array([0.0,1.0]))
    print( "l:", l, "return type", rt, "ID: ", pv_ts )

    rt, l, pv_ts = bg.culc_nearest_distance_to_cross(np.array([0.0,0.0]), np.array([0.0,1.0]))
    print( "l:", l, "return type", rt, "ID: ", pv_ts )

    rt, l, pv_ts = bg.culc_nearest_distance_to_cross(np.array([10.0,0.0]), np.array([0.0,1.0]))
    print( "l:", l, "return type", rt, "ID: ", pv_ts )


    bg_draw = _BoxGarden_draw_test()
    
    bg_draw_boot = BoxGarden_draw.BoxGarden_draw_boot(bg_draw)
    
    cw = circle_wall(r=15.0)
    cw.send_drawobj()
    co = circle_object(np.array([10.0,1.0]), r=3.0)
    co.send_drawobj()
    car = _my_car_obj(np.array([-3.0,-1.0]), rot_theata=3.14/180*10)
    car.send_drawobj()
    
    time.sleep(3)
    car.update_pos_attrs(np.array([3.0,1.0]), rot_theata=3.14/180*90, attrs=[])
    
    time.sleep(3)
    car.restart_pos(np.array([3.0,-1.0]), rot_theata=3.14/180*120)
    
    time.sleep(10)
    bg_draw_boot.finish()
    