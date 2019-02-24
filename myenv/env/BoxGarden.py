# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 15:06:27 2018

@author: alpha
"""

import numpy as np
from enum import Enum


from . import id_generator as idg


class bg_obj_RC(Enum):
    '''
    箱庭オブジェクトに対する交差判定結果を示す列挙型
    '''
    CROSSED = 1       # 始点からの視線ベクトルと交差する
    NOT_CROSSED = 2   # 始点からの視線ベクトルは、交差しない
    IN_OBJECT = 3     # 始点は、物体の中

class bg_object:
    '''
    箱庭オブジェクトの基本クラス。
    箱庭オブジェクトは、子供を木構造で保持することができる。

    交差計算のI/Fを定義するクラス。
    派生クラスはculc_distance_to_cross()をオーバーライドする事
    
    箱庭オブジェクトとしては、点を表すオブジェクトとして実装。
    '''
    def __init__(self, pos=np.zeros((2), dtype=np.float32), arg_rot_th=0.0, parent_id=None):
        '''
        第1引数  2つの要素を持つnumpyの配列。箱庭オブジェクト位置を指定する。
        '''
        self.id = idg._internal_id_singleton.get_new_id()
        self.pid = parent_id
        self.pos = pos.reshape(2,1).astype(np.float32)
        self.rot_th = arg_rot_th
        self.child_list = []
        
    def append_child(self, a_child):
        '''
        子供の以下の要素として追加する処理を行う。
        追加が成功した場合は、Trueを返す
        追加に失敗した場合には、Falseを返す。失敗する場合は、親を持っているが、子供以下に該当する親が見つからなかった場合。
        
        第1引数  子供の以下の要素として追加する対象となるbg_object
        '''
        if a_child.pid is None:
            a_child.pid = self.id
            self.child_list.append(a_child)
            ret = True
        elif a_child.pid == self.id:
            self.child_list.append(a_child)
            ret = True
        else:
            # 深さ優先の再帰検索で親となる箱庭オブジェクトを検索する。
            for in_ch in self.child_list:
                if in_ch.append_child(a_child):
                    ret = True
                    break
            else:
                ret = False
        return ret

    def set_Nrays(self, v1s):
        '''
        交差計算用の視線ベクトルの事前計算
        第1引数  交差計算の視線ベクトルの集合。単位ベクトルでなくてもよい。[2, n]
        '''
        self.vls_size = v1s.shape
        local_v_unit  = v1s / np.linalg.norm(v1s,axis=0)
        local_vm = np.empty(self.vls_size[1]*2*2).reshape(self.vls_size[1],2,2)
        local_vm[:,0,0] =  local_v_unit[0,:]
        local_vm[:,0,1] =  local_v_unit[1,:]
        local_vm[:,1,0] =  local_v_unit[1,:]
        local_vm[:,1,1] = -local_v_unit[0,:]
        self.inv_mat_ray = np.linalg.inv(local_vm)


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
        x = np.zeros(self.vls_size[1])
        ret_type = []
        for idx in range(self.vls_size[1]):
            ret_type.append(BoxGarden.bg_obj_RC.NOT_CROSSED)

        return ret_type, x, ()

    def culc_distance_whole_Nray(self, start_point, v1s, ignore_id=None):
        '''
        自分自身、子、子以下と交差するまでの距離の内最短距離を求める。
        IN_OBJECTが１つでも存在した場合は、IN_OBJECTを返す
        
        第1引数  2つの要素を持つnumpyの配列。交差計算の始点。
        第2引数  2つの要素を持つnumpyの配列。交差計算の視線ベクトル。単位ベクトルでなくてもよい。[2,n]の形
        
        戻り値   ３つの要素を持つリスト
               第1要素   bg_obj_RC型。交差しているかどうかの判定結果
               第2要素   始点からの距離。ただし、交差していない場合(bg_obj_RC.CROSSED以外)は意味を持たない
               第3要素   演算結果の内部情報。基本クラスでは、空のタプル
        補足    基本クラスは、点を表す物体となるため、交差しない。
        '''
        crossed_id = [None for i in range(v1s.shape[1])]
        if self.id == ignore_id:
            ret_type = [bg_obj_RC.NOT_CROSSED for i in range(v1s.shape[1])]
            x        = [-1.0 for i in range(v1s.shape[1])]
        else:
            self.set_Nrays(v1s)
            ret_type, x, a = self.culc_distance_to_cross_Nray(start_point)
            #print('ID: ', self.id, 'PID: ', self.pid, 'ret: ', ret_type)
            for i in range(v1s.shape[1]):
                if ret_type[i] == bg_obj_RC.CROSSED:
                    crossed_id[i] = self.id

        for in_ch in self.child_list:
            ch_ret = in_ch.culc_distance_whole_Nray(start_point, v1s, ignore_id)
            for i in range(v1s.shape[1]):
                if ch_ret[0][i] == bg_obj_RC.IN_OBJECT:
                    ret_type[i] = bg_obj_RC.IN_OBJECT
                elif ch_ret[0][i] == bg_obj_RC.CROSSED:
                    if ret_type[i] == bg_obj_RC.NOT_CROSSED:
                        ret_type[i] = bg_obj_RC.CROSSED
                        x[i] = ch_ret[1][i]
                        crossed_id[i] = ch_ret[2][i]
                    elif ret_type[i] == bg_obj_RC.CROSSED:
                        if x[i] > ch_ret[1][i]:
                            x[i] = ch_ret[1][i]
                            crossed_id[i] = ch_ret[2][i]
                    else:   # ret_type[i] == bg_obj_RC.IN_OBJECT:
                        pass
        
        return ret_type, x, crossed_id
    
    def update(self):
        pass

class BoxGarden:
    '''
    箱庭を表すクラス
    '''
    def __init__(self):
        '''
        第1引数  
        第2引数  
        '''
        self.bg_objs = []

    def append_bg_obj(self, bgo):
        '''
        箱庭に箱庭オブジェクトを追加する。
        追加する箱庭オブジェクトのpidがNoneではない場合、親の箱庭オブジェクトを検索の上、箱庭オブジェクトの子として追加する
        
        第1引数  追加する対象となるbg_object
        '''
        if bgo.pid is None:
            self.bg_objs.append(bgo)
        else:
            for in_bgo in self.bg_objs:
                if in_bgo.append_child(bgo):
                    break
            else:
                self.bg_objs.append(bgo)

    def culc_nearest_distance_to_cross_Nray(self, start_point, v1s, ignore_id=None):
        '''
        箱庭内の箱庭オブジェクト、それぞれの子、子以下と交差するまでの距離の内、最短距離を求める。
        IN_OBJECTが１つでも存在した場合は、IN_OBJECTを返す
        
        第1引数  2つの要素を持つnumpyの配列。交差計算の始点。
        第2引数  2つの要素を持つnumpyの配列。交差計算の視線ベクトル。単位ベクトルでなくてもよい。[2,n]の形
        
        戻り値   ３つの要素を持つリスト
               第1要素   bg_obj_RC型。交差しているかどうかの判定結果
               第2要素   始点からの距離。ただし、交差していない場合(bg_obj_RC.CROSSED以外)は意味を持たない。n個の要素を持つnumpy配列
               第3要素   演算結果の内部情報。基本クラスでは、空のタプル
        補足    基本クラスは、点を表す物体となるため、交差しない。
        '''
        ret_type = [bg_obj_RC.NOT_CROSSED for i in range(v1s.shape[1])]
        x        = [-1.0 for i in range(v1s.shape[1])]
        crossed_id = [None for i in range(v1s.shape[1])]
        for in_ch in self.bg_objs:
            if in_ch.id == ignore_id:
                continue
            
            ch_ret = in_ch.culc_distance_whole_Nray(start_point, v1s, ignore_id)
            for i in range(v1s.shape[1]):
                if ch_ret[0][i] == bg_obj_RC.IN_OBJECT:
                    ret_type[i] = bg_obj_RC.IN_OBJECT
                elif ch_ret[0][i] == bg_obj_RC.CROSSED:
                    if ret_type[i] == bg_obj_RC.NOT_CROSSED:
                        ret_type[i] = bg_obj_RC.CROSSED
                        x[i] = ch_ret[1][0,i]
                        crossed_id[i] = ch_ret[2][i]
                    elif ret_type[i] == bg_obj_RC.CROSSED:
                        if x[i] > ch_ret[1][0,i]:
                            x[i] = ch_ret[1][0,i]
                            crossed_id[i] = ch_ret[2][i]
                    else:   # ret_type[i] == bg_obj_RC.IN_OBJECT:
                        pass

        return ret_type, x, crossed_id

if __name__ == '__main__':
    bg = BoxGarden()
    bgo_a = bg_object(parent_id=None)
    bgo_b = bg_object(parent_id=None)
    bgo_c = bg_object(parent_id=None)
    bgo_d = bg_object(parent_id=None)
    bgo_e = bg_object(parent_id=None)
    bgo_f = bg_object(parent_id=None)
    
    bg.append_bg_obj(bgo_a)
    bg.append_bg_obj(bgo_b)
    bg.append_bg_obj(bgo_c)

    bgo_d.pid = bgo_b.id
    bgo_e.pid = bgo_b.id
    bgo_f.pid = bgo_e.id
    
    bg.append_bg_obj(bgo_d)
    bg.append_bg_obj(bgo_e)
    bg.append_bg_obj(bgo_f)
    
    print(bg)
    
    print(bgo_a)
    print(bgo_b)
    print(bgo_c)
    print(bgo_d)
    print(bgo_e)
    print(bgo_f)
    
    ret = bg.culc_nearest_distance_to_cross(np.zeros((2), dtype=np.float32), np.ones((2), dtype=np.float32))
    print(ret)
    