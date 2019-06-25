# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 15:06:27 2018

@author: alpha
"""

import numpy as np
#import math
from enum import Enum

import logging

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FFMpegWriter

import os
import multiprocessing as mp
import time
import threading

from . import id_generator as idg
from . import drawing_trace2 as dt2

#import BoxGarden

class bg_object_draw():
    '''
    bg_objectとDraw機能を結びつけるクラス
    '''
    def __init__(self):
        self.id = idg._internal_id_singleton.get_new_id()

    def process_msg(self, msg, bg_draw):
        pass

    def process_add(self, msg, bg_draw):
        pass

    def process_del(self, msg, bg_draw):
        pass

    def process_update(self, msg, bg_draw):
        pass

    def update_subplot(self, data):
        pass

class bg_msg_ID(Enum):
    '''
    メッセージID
    '''
    INVALID = -1
    START = 1
    FINISH = 2
    COMMON = 3
    ADD = 4
    DELETE = 5
    UPDATE = 6
    START_VIDEO_RECORD = 7
    FINISH_VIDEO_RECORD = 8

class bg_message_base():
    '''
    BoxGardenとDraw機能の間をつなぐメッセージの基本クラス
    '''
    def __init__(self, mid=bg_msg_ID.INVALID, target_id=None):
        '''
            第1引数  message id
        '''
        self.mid = mid
        self.tid = target_id

    def command_ADD(self, a_obj):
        '''
            BoxGarden_drawへオブジェクトを追加する

            第1引数  BoxGarden_drawへ追加するオブジェクト
        '''
        self.append_obj = a_obj
        self.mid = bg_msg_ID.ADD
        self.tid = a_obj.id

    def command_DELETE(self, target_id):
        '''
            BoxGarden_drawからオブジェクトを削除する

            第1引数  削除するオブジェクトのID
        '''
        self.tid = target_id
        self.mid = bg_msg_ID.DELETE

    def command_UPDATE(self, target_id, attrs={}):
        '''
            BoxGarden_drawからオブジェクトを削除する

            第1引数  削除するオブジェクトのID
        '''
        self.tid = target_id
        self.mid = bg_msg_ID.UPDATE
        self.updated_attrs = attrs

    def command_START_VIDEO_REC(self, fname):
        '''
            matplotlibのfigの録画を開始する

            第1引数  録画したファイルの出力先
        '''
        self.mid = bg_msg_ID.START_VIDEO_RECORD
        self.video_fname = fname

    def command_FINISH_VIDEO_REC(self):
        '''
            matplotlibのfigの録画を終了する
        '''
        self.mid = bg_msg_ID.FINISH_VIDEO_RECORD

class BoxGarden_draw():
    '''
    BoxGardenとDraw機能を結びつけるクラス
    '''
    def __init__(self, subplot_num=[1,1]):
        '''
            第1引数  2つの要素を持つリスト。第一要素が縦のグラフの数、第2要素が横のグラフの数
        '''
        self.subplot_num = subplot_num
        self.scn = None
        self.bg_obj_list = {}
        self.mq_to_rcv = None       # レシーバー側へメッセージを送るためのメッセージキュー
        self.mq_f_client = None     # クライアント側からのメッセージを受け取るためのメッセージキュー
        self.mq_to_scn = None       # matplotの制御スレッドへメッセージを送るためのメッセージキュー
        self.access_lock = None     # メッセージ処理スレッドとmatplot制御スレッドとの排他制御用ロック
        self.updated_marker = False
    
    # -------------------------
    # Sender side APIs
    def setup_sender(self, mq_to_rcv=None):
        if mq_to_rcv is not None:
            self.mq_to_rcv = mq_to_rcv   # レシーバー側へメッセージを送るためのメッセージキュー

    def send_msg(self, msg):
        self.mq_to_rcv.put(msg)

    # -------------------------
    # Receiver side APIs
    def setup_receiver(self, mq_f_client=None, mq_to_scn=None, ac_lock=None):
        if mq_f_client is not None:
            self.mq_f_client = mq_f_client   # クライアント側からのメッセージを受け取るためのメッセージキュー
        if mq_to_scn is not None:
            self.mq_to_scn = mq_to_scn     # matplotの制御スレッドへメッセージを送るためのメッセージキュー
        if ac_lock is not None:
            self.access_lock = ac_lock     # メッセージ処理スレッドとmatplot制御スレッドとの排他制御用ロック


    def start_screen(self):
        self.scn = dt2.screen(self.subplot_num)
        self.updated_marker = True


    def process_msg(self, msg):
        if msg.mid == bg_msg_ID.ADD:
            if msg.tid is not None:
                if msg.tid in self.bg_obj_list.keys():
                    print('bg_msg_ID.ADD: tid ', msg.tid, ' is already existed. Delete old at first.')
                    self.bg_obj_list[msg.tid].process_add(msg, self)
                print('bg_msg_ID.ADD: tid ', msg.tid)
                self.bg_obj_list[msg.tid] = msg.append_obj
                self.bg_obj_list[msg.tid].process_add(msg, self)
            else:
                print('Error: tid is None for bg_msg_ID.ADD')
        else:
            if  msg.tid in self.bg_obj_list.keys():
                if msg.mid == bg_msg_ID.COMMON:
                    self.bg_obj_list[msg.tid].process_msg(msg, self)
                elif msg.mid == bg_msg_ID.UPDATE:
                    self.bg_obj_list[msg.tid].process_update(msg, self)
                elif msg.mid == bg_msg_ID.DELETE:
                    self.bg_obj_list[msg.tid].process_del(msg, self)
                    del self.bg_obj_list[msg.tid]
                else:
                    print('Error: unknown MID is Error:', msg.mid)
            else:
                print('Error: There is no ID:  mid: ', msg.mid, '  tid: ', msg.tid)

    def update_subplot(self, data):
        for bgo in self.bg_obj_list.values():
            bgo.update_subplot(data)

        self.scn.update_subplot()

class RecordingStatus(Enum):
    '''
    録画状態を示す状態ID
    '''
    INVALID = -1
    WAITING_FOR_FIG = 1
    VIDEO_RECORDING = 2

def prepare_video_dir(path_name):
    print('Requested video file name: ', path_name)
    base_dir_pair = os.path.split(path_name)
    print('Splited video file name: ', base_dir_pair)

    os.makedirs(base_dir_pair[0], exist_ok=True)


def dt2_show(bg_draw, update_interval):
    print("dt2_show:", update_interval)
    metadata = dict(title='bg_ai_test', artist='T.A',
                comment='Movie log')

    fps = int(1000/update_interval)
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    def gen():
        continue_flag = True
        video_fname = None
        i = 0
        recording_status = RecordingStatus.INVALID
        
        while continue_flag:
            if bg_draw.mq_to_scn.empty() != True:
                msg = bg_draw.mq_to_scn.get()
                if msg.mid == bg_msg_ID.START:
                    print('command bg_msg_ID.START is recieved.')
                elif msg.mid == bg_msg_ID.FINISH:
                    print('command bg_msg_ID.FINISH is recieved.')
                    if recording_status == RecordingStatus.VIDEO_RECORDING:
                        print('Stop to record the video')
                        writer.finish()
                    recording_status = RecordingStatus.INVALID
                    continue_flag = False
                elif msg.mid == bg_msg_ID.START_VIDEO_RECORD:
                    if recording_status == RecordingStatus.VIDEO_RECORDING:
                        writer.finish()
                    recording_status = RecordingStatus.INVALID

                    if msg.video_fname is not None:
                        video_fname = msg.video_fname
                        prepare_video_dir(msg.video_fname)
                        if bg_draw.scn is not None:
                            print('Start to record the video')
                            with bg_draw.access_lock:
                                writer.setup(bg_draw.scn.fig, video_fname)
                            recording_status = RecordingStatus.VIDEO_RECORDING
                        else:
                            print('Waiting for fig to record the video')
                            recording_status = RecordingStatus.WAITING_FOR_FIG
                    else:
                        print('No video file name. Give up to record the video')
                elif msg.mid == bg_msg_ID.FINISH_VIDEO_RECORD:
                    if recording_status == RecordingStatus.VIDEO_RECORDING:
                        print('Stop to record the video')
                        writer.finish()
                    recording_status = RecordingStatus.INVALID
                else:
                    print('dt2_show() unhandles the message: ', msg)

            if recording_status == RecordingStatus.WAITING_FOR_FIG:
                if bg_draw.scn is not None:
                    print('Release the waiting status and Start to record the video')
                    with bg_draw.access_lock:
                        writer.setup(bg_draw.scn.fig, video_fname)
                    recording_status = RecordingStatus.VIDEO_RECORDING

            #print("yield in gen:", i, recording_status)

            yield (i, recording_status)
            i = i + 1

        print('dt2_show() received ', continue_flag)

        print('Exit dt2_show().')

    def func(data):
#        logger.info("i = %d", data)
        #print("anime frame:", data)

        with bg_draw.access_lock:
            if bg_draw.updated_marker:
                bg_draw.update_subplot(data[0])
    
                if data[1] == RecordingStatus.VIDEO_RECORDING:
                    writer.grab_frame()
            
            bg_draw.updated_marker = False


    # animation.FuncAnimation()の戻り値を保持する変数を定義しておかないと、funcの呼び出しが開始されない。GCでオブジェクトが捨てられてしまう場合があるのかもしれない。
    with bg_draw.access_lock:
        ani = animation.FuncAnimation(bg_draw.scn.fig,
                                      func, frames=gen, blit=False, interval=update_interval, repeat=False)

    #ani.save("cycloid.mp4")
    plt.show()

def message_loop(bg_draw):
    while True:
        msg = bg_draw.mq_f_client.get()
        with bg_draw.access_lock:
            bg_draw.updated_marker = True
            if msg.mid == bg_msg_ID.START:
                bg_draw.mq_to_scn.put(msg)
                print('command bg_msg_ID.START pass to anime')
            elif msg.mid == bg_msg_ID.FINISH:
                bg_draw.mq_to_scn.put(msg)
                break
            elif msg.mid == bg_msg_ID.START_VIDEO_RECORD:
                bg_draw.mq_to_scn.put(msg)
            elif msg.mid == bg_msg_ID.FINISH_VIDEO_RECORD:
                bg_draw.mq_to_scn.put(msg)
            else:
                bg_draw.process_msg(msg)

def start_draw_side(bg_draw, mq, update_interval):
    bg_draw.setup_receiver( mq_f_client=mq, mq_to_scn=mp.Queue(), ac_lock=threading.Lock())
    bg_draw.start_screen()  # プロセス起動後に、dt2.screenを開始する。同一プロセス空間内で起動を行うため。
    thread_1 = threading.Thread(target=message_loop, args=(bg_draw,))
    thread_1.start()

    dt2_show(bg_draw, update_interval)


class BoxGarden_draw_boot():
    def __init__(self, bg_draw, update_interval=200):
        '''
            第1引数  描画制御対象となるBoxGarden_draw
            第2引数  描画更新間隔[msec]
        '''
        mq = mp.Queue()
        self.target_bg_draw = bg_draw
        self.p = mp.Process(target=start_draw_side, args=(bg_draw,mq,update_interval,))
        self.p.start()

        self.target_bg_draw.setup_sender(mq_to_rcv=mq)
        mes = bg_message_base(bg_msg_ID.START)
        self.target_bg_draw.send_msg(mes)

    def finish(self):
        mes = bg_message_base(bg_msg_ID.FINISH)
        self.target_bg_draw.send_msg(mes)
        self.p.join()


# 以下は、テスト用コード

class _bg_draw_obj_test_my_car(bg_object_draw):
    class my_car_msg_ID(Enum):
        '''
        メッセージID
        '''
        INVALID = -1
        START = 1
        FINISH = 2
        RESET = 3

    class my_car_msg(bg_message_base):
        def __init__(self, target_id):
            super().__init__(mid=bg_msg_ID.COMMON, target_id=target_id)
            self.my_car_cmd = _bg_draw_obj_test_my_car.my_car_msg_ID.INVALID

    def __init__(self):
        super().__init__()
        self.pre_data = None
        self.offset_backup = None

    def process_add(self, msg, bg_draw):
        ax = bg_draw.scn.subplot_list[0].ax

        self.my_car = dt2.drawobj_empty()
        dt2.drawobj_triangle(ax, 0.0, 0.0, -0.5, -1.0, 0.5, -1.0, parent=self.my_car)
        dt2.drawobj_line(ax, 0.0, 0.0, 0.0, 2.0, parent=self.my_car)
        dt2.drawobj_line(ax, 0.0, 0.0,  1.0, 1.0, parent=self.my_car)
        dt2.drawobj_line(ax, 0.0, 0.0, -1.0, 1.0, parent=self.my_car)
        dt2.drawobj_circle_point(ax, 0.0, 1.0, parent=self.my_car)
        dt2.drawobj_circle(ax, 0.0, 0.0, 2.0, parent=self.my_car)
        self.line = dt2.drawobj_poly(ax)
        self.line.append(0.0, 0.0)
        bg_draw.scn.subplot_list[0].append_drawobj(self.my_car)
        bg_draw.scn.subplot_list[0].append_drawobj(self.line)

    def process_msg(self, msg, bg_draw):
        if msg.my_car_cmd is not None:
            if msg.my_car_cmd == _bg_draw_obj_test_my_car.my_car_msg_ID.RESET:
                if self.pre_data is not None:
                    self.offset_backup = self.pre_data


    def update_subplot(self, data):
        self.pre_data = data
        if self.offset_backup is not None:
            step = data - self.offset_backup
        else:
            step = data

        self.my_car.move[0] = step /20.0
        self.my_car.move[1] = step /20.0
        self.my_car.rot_theata = step/180.0 * np.pi

        self.line.append(self.my_car.move[0], self.my_car.move[1])

class _bg_draw_obj_test_line(bg_object_draw):
    def process_add(self, msg, bg_draw):
        ax2 = bg_draw.scn.subplot_list[1].ax

        self.line2 = dt2.drawobj_poly(ax2, attr='r-')
        self.line2.append( 0.0, 0.0)
        self.line2.append( 1.0, 1.0)
        self.line2.append(-1.0, 2.0)
        self.line2.append( 1.0, 3.0)
        self.line2.append(-1.0, 4.0)
        self.line2.append( 1.0, 5.0)
        self.line2.append(-1.0, 6.0)

        bg_draw.scn.subplot_list[1].append_drawobj(self.line2)

    def update_subplot(self, data):
        self.line2.rot_theata = 2.0 * data/180.0 * np.pi

class _BoxGarden_draw_test(BoxGarden_draw):
    def __init__(self):
        super().__init__([1,2])

    def start_screen(self):
        super().start_screen()

        self.scn.subplot_list[0].ax.set_xlim(-10, 10)
        self.scn.subplot_list[0].ax.set_ylim(-10, 10)
        self.scn.subplot_list[0].ax.set_xlabel('x')
        self.scn.subplot_list[0].ax.set_ylabel('y')
        self.scn.subplot_list[0].ax.grid()

        self.scn.subplot_list[1].ax.set_xlim(-20, 20)
        self.scn.subplot_list[1].ax.set_ylim(-20, 20)
        self.scn.subplot_list[1].ax.set_xlabel('x')
        self.scn.subplot_list[1].ax.set_ylabel('y')



if __name__ == '__main__':
    mp.freeze_support()

    logging.basicConfig(format='%(asctime)s/%(module)s/%(name)s/%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
#    drawobj_base.debug_logger.setLevel(logging.CRITICAL)


    bg_draw = _BoxGarden_draw_test()
    bgdo_my_car = _bg_draw_obj_test_my_car()
    bgdo_line2 = _bg_draw_obj_test_line()

    bg_draw_boot = BoxGarden_draw_boot(bg_draw)

    mes = bg_message_base()
    mes.command_ADD(bgdo_my_car)
    bg_draw.send_msg(mes)

    mes = bg_message_base()
    mes.command_START_VIDEO_REC('test/test_demo1.mp4')
    bg_draw.send_msg(mes)

    time.sleep(5)

    mes = _bg_draw_obj_test_my_car.my_car_msg(bgdo_my_car.id)
    mes.my_car_cmd = _bg_draw_obj_test_my_car.my_car_msg_ID.RESET
    bg_draw.send_msg(mes)

    time.sleep(5)

    mes = bg_message_base()
    mes.command_FINISH_VIDEO_REC()
    bg_draw.send_msg(mes)
    mes = bg_message_base()
    mes.command_START_VIDEO_REC('test/test_demo2.mp4')
    bg_draw.send_msg(mes)

    mes = bg_message_base()
    mes.command_ADD(bgdo_line2)
    bg_draw.send_msg(mes)

    time.sleep(5)

    mes = bg_message_base()
    mes.command_DELETE(bgdo_line2.id)
    bg_draw.send_msg(mes)

    time.sleep(5)
    bg_draw_boot.finish()
