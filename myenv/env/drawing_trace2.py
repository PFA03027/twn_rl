# -*- coding: utf-8 -*-
"""
Created on Fri May  5 19:58:44 2017

@author: alpha
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import multiprocessing as mp
import time
import copy
from enum import Enum

from . import id_generator as idg

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
#logger.setLevel(logging.INFO)
#handler = logging.StreamHandler()
#handler.setLevel(logging.NOTSET)
#logger.addHandler(handler)

logger.info('Start logger of drawing_trace2...')


class drawobj_base():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARNING)
    debug_logger = logging.getLogger(__name__)
#    debug_logger.setLevel(logging.DEBUG)

    @classmethod
    def get_rot_mat(cls, r):
        return np.matrix( (( math.cos(r), -math.sin(r)),
                           ( math.sin(r),  math.cos(r)) ), dtype=np.float32 )

    def get_xx_yy(self):
        '''
            [仮想関数] 変換対象座標情報を具象クラスから得る
        '''
        drawobj_base.logger.error("No concrate class of drawobj set_data")
        return [],[]

    def set_data(self):
        '''
            [仮想関数] 変換後の座標情報を具象クラスのデータに設定する
        '''
        drawobj_base.logger.error("No concrate class of drawobj set_data")
        return

    def __init__( self, parent=None, plot_obj=None ):
        self.parent = parent    # 親オブジェクトへの参照
        self.child = []         # 子オブジェクトへの参照の集合
        self.id = idg._internal_id_singleton.get_new_id()
        if parent is not None:
            parent.child.append(self)

        self.xx = []            # plot()のx軸座標情報の集合
        self.yy = []            # plot()のy軸座標情報の集合
        self.plot_obj = plot_obj
        self.color = None
        self.line_style = None

        # y = scale * (rotmax * xx or yy) + move
        self.rot_theata = 0.0
        self.move = np.zeros(2, dtype=np.float32).reshape(2,1)
        self.scale = 1.0

        self.effective_rot_theata = None
        self.effective_move = None
        self.effective_scale = None
        self.effective_xx = None
        self.effective_yy = None
        
        self.enable_trans = True

        
    def calc_effective_param(self):
        '''
            上位層から座標変換を伝搬させる関数
            上位層の座標変換と、自身の座標変換の合成変換の結果を、effective_****に反映する。
            反映式は、下記の通り。
            # z = p.scale * (p.rotmax * y) + p.move
            # z = p.scale * (p.rotmax * (scale * (rotmax * xx or yy) + move)) + p.move
            # z = p.scale * (p.rotmax * scale * (rotmax * xx or yy) + p.rotmax * move) + p.move
            # z = (p.scale * scale) * (p.rotmax * rotmax) * xx or yy + (p.scale * p.rotmax * move + p.move)
        '''
        # 上位層の情報の反映
        if self.parent is None:
            self.effective_rot_theata = copy.copy(self.rot_theata)
            self.effective_move = copy.copy(self.move).reshape(2,1)
            self.effective_scale = copy.copy(self.scale)
        else:
            self.effective_rot_theata = self.parent.effective_rot_theata + self.rot_theata
            self.effective_move = self.parent.effective_scale * drawobj_base.get_rot_mat(self.parent.effective_rot_theata).dot(self.move) + self.parent.effective_move
            self.effective_scale = self.parent.effective_scale * self.scale

        # 自身の描画オブジェクトの座標情報の変換
        lxx, lyy = self.get_xx_yy()
        self.xx = lxx.copy()
        self.yy = lyy.copy()
        if self.enable_trans:
            data = np.array((self.xx, self.yy), dtype=np.float32).reshape(2,len(self.yy))
            nrow, ncol = data.shape
            data = self.effective_scale * drawobj_base.get_rot_mat(self.effective_rot_theata).dot(data)
            offset = np.empty(nrow*ncol).reshape(nrow,ncol)
            offset[0,:] = self.effective_move[0,0]
            offset[1,:] = self.effective_move[1,0]
            data = data + offset
            self.effective_xx = data[0,:]
            self.effective_yy = data[1,:]
        else:
            self.effective_xx = self.xx
            self.effective_yy = self.yy
        
        # 描画情報を反映する
        if (self.plot_obj is not None) and (self.color is not None):
            self.plot_obj.set_color(self.color)
            self.color = None
        if (self.plot_obj is not None) and (self.line_style is not None):
            self.plot_obj.set_color(self.line_style)
            self.line_style = None
        self.set_data()

        # 子オブジェクトへ演算を伝搬させる
        for i_do in self.child:
            i_do.calc_effective_param()
    
    def set_color(self, color):
        '''
            Please see set_color() of matplotlib.lines.Line2D
            https://matplotlib.org/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D
        '''
        self.color = color

    def set_line_style(self, style):
        '''
            Please see set_color() of matplotlib.lines.Line2D
            https://matplotlib.org/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D
        '''
        self.line_style = style


class drawobj_empty(drawobj_base):
    '''
        なにも描画しないコンテナとしての描画オブジェクト
    '''
    def __init__( self, parent=None ):
        super().__init__(parent=parent)

    def get_xx_yy(self):
        '''
            空の座標情報を返す
        '''
        drawobj_base.debug_logger.info("drawobj_empty.get_xx_yy")
        return [0.0], [0.0]

    def set_data(self):
        '''
            なにもしない
        '''
        return

class drawobj_circle_point(drawobj_base):
    '''
        小さい円で点を表す描画オブジェクト
    '''
    def __init__( self, ax, x, y, attr='bo', parent=None ):
        self.x = [x]
        self.y = [y]
        self.point, = ax.plot(self.x, self.y, attr, ms=4)
        super().__init__(parent=parent, plot_obj=self.point)

    def update( self, x, y ):
        self.x = [x]
        self.y = [y]

    def get_xx_yy(self):
        '''
            点の変換対象座標情報を返す
        '''
        drawobj_base.debug_logger.info("drawobj_circle_point.get_xx_yy")
        return self.x, self.y

    def set_data(self):
        '''
            点の座標を設定する
        '''
        self.point.set_data(self.effective_xx, self.effective_yy)
        return

class drawobj_line(drawobj_base):
    '''
        2点間を結ぶ直線を表す描画オブジェクト
    '''
    def __init__( self, ax, x1, y1, x2, y2, attr='y-', parent=None ):
        self.x = [x1, x2]
        self.y = [y1, y2]
        self.line, = ax.plot(self.x, self.y, attr, ms=4)
        super().__init__(parent=parent, plot_obj=self.line)

    def update( self, x1, y1, x2, y2 ):
        self.x = [x1, x2]
        self.y = [y1, y2]

    def get_xx_yy(self):
        '''
            直線の変換対象座標情報を返す
        '''
        drawobj_base.debug_logger.info("drawobj_line.get_xx_yy")
        return self.x, self.y

    def set_data(self):
        '''
            直線の座標を設定する
        '''
        self.line.set_data(self.effective_xx, self.effective_yy)
        return

class drawobj_triangle(drawobj_base):
    '''
        三角形を表す描画オブジェクト
    '''
    def __init__( self, ax, x1, y1, x2, y2, x3, y3, attr='y-', parent=None ):
        self.x = [x1, x2, x3, x1]
        self.y = [y1, y2, y3, y1]
        self.triangle, = ax.plot(self.x, self.y, attr, ms=4)
        super().__init__(parent=parent, plot_obj=self.triangle)
        
    def update(self, x1, y1, x2, y2, x3, y3):
        self.x = [x1, x2, x3, x1]
        self.y = [y1, y2, y3, y1]

    def get_xx_yy(self):
        '''
            三角形の変換対象座標情報を返す
        '''
        drawobj_base.debug_logger.info("drawobj_triangle.get_xx_yy")
        return self.x, self.y

    def set_data(self):
        '''
            三角形の座標を設定する
        '''
        self.triangle.set_data(self.effective_xx, self.effective_yy)
        return

class drawobj_poly(drawobj_base):
    '''
        複数点間を結ぶポリゴン描画オブジェクト
    '''
    def __init__( self, ax, attr='g--', parent=None ):
        self.x = []
        self.y = []
        self.poly, = ax.plot(self.x, self.y, attr)
        super().__init__(parent=parent, plot_obj=self.poly)
        
    def append(self, x, y):
        self.x.append(x)
        self.y.append(y)

    def clear(self):
        self.x = []
        self.y = []

    def get_xx_yy(self):
        '''
            ポリゴンの変換対象座標情報を返す
        '''
        drawobj_base.debug_logger.info("drawobj_poly.get_xx_yy")
        return self.x, self.y

    def set_data(self):
        '''
            ポリゴンの座標を設定する
        '''
        self.poly.set_data(self.effective_xx, self.effective_yy)
        return

class drawobj_circle(drawobj_base):
    '''
        円を表す描画オブジェクト
    '''
    def circle(self, a, b, r):
        # 点(a,b)を中心とする半径rの円
        T = 17
        x, y = [0]*T, [0]*T
        for i,theta in enumerate(np.linspace(0,2*np.pi,T)):
            x[i] = a + r*np.cos(theta)
            y[i] = b + r*np.sin(theta)
        return x, y

    def __init__( self, ax, x1, y1, R, attr='g--', parent=None ):
        self.x, self.y = self.circle(x1, y1, R)
        self.circle_poly, = ax.plot(self.x, self.y, attr, ms=4)
        super().__init__(parent=parent, plot_obj=self.circle_poly)
        
    def update(self, a, b, r):
        self.x, self.y = self.circle(a, b, r)
        
    def get_xx_yy(self):
        '''
            円の変換対象座標情報を返す
        '''
        drawobj_base.debug_logger.info("drawobj_circle.get_xx_yy")
        return self.x, self.y

    def set_data(self):
        '''
            円の座標を設定する
        '''
        self.circle_poly.set_data(self.effective_xx, self.effective_yy)
        return

class drawobj_arc(drawobj_base):
    '''
        円弧を表す描画オブジェクト
    '''
    def arc(self, a, b, r, s_angle, e_angle):
        # 点(a,b)を中心とする半径rの円
        T = 17
        x, y = [0]*T, [0]*T
        for i,theta in enumerate(np.linspace(s_angle, e_angle,T)):
            x[i] = a + r*np.cos(theta)
            y[i] = b + r*np.sin(theta)
        return x, y

    def __init__( self, ax, x1, y1, R, s_angle=0.0, e_angle=2*np.pi, attr='g--', parent=None ):
        self.x, self.y = self.arc(x1, y1, R, s_angle, e_angle)
        self.arc_poly, = ax.plot(self.x, self.y, attr, ms=4)
        super().__init__(parent=parent, plot_obj=self.arc_poly)
        
    def update(self, a, b, r, s_angle, e_angle):
        self.x, self.y = self.arc(a, b, r, s_angle, e_angle)
        
    def get_xx_yy(self):
        '''
            円の変換対象座標情報を返す
        '''
        drawobj_base.debug_logger.info("drawobj_circle.get_xx_yy")
        return self.x, self.y

    def set_data(self):
        '''
            円の座標を設定する
        '''
        self.arc_poly.set_data(self.effective_xx, self.effective_yy)
        return


class screen_subplot():
    '''
        matplotの１つのグラフに対する情報や、描画の機能を提供する。
        このクラスに紐づけられるdrawobj_baseが描画の対象となる表示物で、複数を登録可能
        また、drawobj_baseが再帰構造による木構造をとる事ができるため、描画物の階層構造も対応可能
    '''
    def __init__(self, ax, arg_pre_update_func=None, arg_post_update_func=None):
        '''
            subplotでの描画の初期化

            第1引数  subplotへの参照
        '''
        self.ax = ax
        self.drawobj_top_list = []
        self.pre_update_func  = arg_pre_update_func
        self.post_update_func = arg_post_update_func
        
    def append_drawobj(self, do):
        '''
            subplotを親とするdrawobj_baseの追加

            第1引数  drawobj_baseへの参照
        '''
        self.drawobj_top_list.append(do)

    def update_drawobj(self):
        '''
            描画オブジェクトの更新処理
        '''
        if self.pre_update_func is not None:
            self.pre_update_func(self.ax)

        for i_do in self.drawobj_top_list:
            i_do.calc_effective_param()

        if self.post_update_func is not None:
            #self.post_update_func(self.ax)
            self.ax.set_xlim(left=0.0, auto=True)
            self.ax.set_ylim(auto=True)

class screen():
    '''
        matplotでの描画機能を提供するクラス
        Windowの中に複数のグラフ描画領域を持つ
    '''
    def __init__(self, subplot_num, window_title=None):
        '''
            matplotでの描画の初期化
            subplotをによる複数のグラフを準備できる。

            第1引数  2つの要素を持つリスト。第一要素が縦のグラフの数、第2要素が横のグラフの数
        '''
        self.id = idg._internal_id_singleton.get_new_id()
        self.fig = plt.figure(figsize=(10, 6))
        if window_title is None:
            self.fig.canvas.set_window_title('default title')
        else:
            self.fig.canvas.set_window_title(window_title)
        self.subplot_list = []
        subplot_num_count = 1
        for l in range(0,subplot_num[0]):   # 縦のループ
            for n in range(0,subplot_num[1]):   # 横のループ
                ax = self.fig.add_subplot(subplot_num[0], subplot_num[1], subplot_num_count)
                ax.set_ylim(-10, 10)
                ax.set_xlim(-10, 10)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.grid()
                self.subplot_list.append( screen_subplot(ax) )
                subplot_num_count += 1

    def update_subplot(self):
        '''
            描画オブジェクトの更新処理
        '''
        for i_sp in self.subplot_list:
            i_sp.update_drawobj()


if __name__ == '__main__':
    mp.freeze_support()

    logging.basicConfig(format='%(asctime)s/%(module)s/%(name)s/%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    drawobj_base.debug_logger.setLevel(logging.CRITICAL)

    scn = screen([1,2])
    scn.subplot_list[0].ax.set_xlim(-10, 10)
    scn.subplot_list[0].ax.set_ylim(-10, 10)
    scn.subplot_list[0].ax.set_xlabel('x')
    scn.subplot_list[0].ax.set_ylabel('y')
    scn.subplot_list[0].ax.grid()
    
    ax = scn.subplot_list[0].ax
    
    my_car = drawobj_empty()
    my_car_p1 = drawobj_triangle(ax, 0.0, 0.0, -0.5, -1.0, 0.5, -1.0, parent=my_car)
    my_car_p2 = drawobj_line(ax, 0.0, 0.0, 0.0, 2.0, parent=my_car)
    my_car_p3 = drawobj_line(ax, 0.0, 0.0,  1.0, 1.0, parent=my_car)
    my_car_p4 = drawobj_line(ax, 0.0, 0.0, -1.0, 1.0, parent=my_car)
    my_car_p5 = drawobj_circle_point(ax, 0.0, 1.0, parent=my_car)
    my_car_p6 = drawobj_circle(ax, 0.0, 0.0, 2.0, parent=my_car)
    line = drawobj_poly(ax)
    line.append(0.0, 0.0)
    scn.subplot_list[0].append_drawobj(my_car)
    scn.subplot_list[0].append_drawobj(line)

    ax2 = scn.subplot_list[1].ax
    scn.subplot_list[1].ax.set_xlim(-20, 20)
    scn.subplot_list[1].ax.set_ylim(-20, 20)
    scn.subplot_list[1].ax.set_xlabel('x')
    scn.subplot_list[1].ax.set_ylabel('y')
    
    line2 = drawobj_poly(ax2, attr='r-')
    line2.append( 0.0, 0.0)
    line2.append( 1.0, 1.0)
    line2.append(-1.0, 2.0)
    line2.append( 1.0, 3.0)
    line2.append(-1.0, 4.0)
    line2.append( 1.0, 5.0)
    line2.append(-1.0, 6.0)

    scn.subplot_list[1].append_drawobj(line2)


    def gen():
        for i in range(100):
            yield i

    def func(data):
        logger.info("i = %d", data)

        my_car.move[0] = data /20.0
        my_car.move[1] = data /20.0
        my_car.rot_theata = data/180.0 * np.pi
        
        line.append(my_car.move[0], my_car.move[1])

        line2.rot_theata = 2.0 * data/180.0 * np.pi
        
        scn.update_subplot()

    ani = animation.FuncAnimation(scn.fig,
            func, gen, blit=False, interval=100, repeat=False)

    #ani.save("cycloid.mp4")
    plt.show()

    