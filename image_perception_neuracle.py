# -*- coding: utf-8 -*-

import sys
import os

# 获取当前脚本文件的路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 将上一级目录添加到 sys.path
parent_dir = os.path.join(script_dir, '..')
sys.path.append(parent_dir)

import numpy as np
import pygame as pg

from utils.task import Task

import argparse
from neuracle_lib.triggerBox import TriggerBox,PackageSensorPara

from neuracle_lib.dataServer import DataServerThread
import time
import csv
import threading

def save_matrix_to_csv(matrix, filename):
    with open(filename, 'a+', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(matrix)


class MotionImg(Task):
    def __init__(self, images=None, tasks=None, type=1, num_per_event=10):
        super().__init__(exp_name='image')
        self.mosaic = pg.image.load('image/mosaic.jpg')
        self.mosaic = pg.transform.scale(self.mosaic, (1200, 900))
        self.bg_color = (0, 0, 0)
        self.font_color = (255, 255, 255)
        self.type = type
        self.num_per_event = num_per_event
        self.tasks = tasks
        self.code_book = dict(zip(self.tasks, np.arange(1, 1 + 2 * len(self.tasks), 2)))
        self.cross = pg.font.Font(self.default_font, self.resize_value(200)).render('+', True, 'white')
        self.rect = pg.Surface((1220, 920))
        self.rect.fill(self.bg_color)
        self.images = images
        self.logger(str(self.code_book))
        global count
        count = 0
        # 配置设备

        # neuracle = dict(device_name='Neuracle', hostname='127.0.0.1', port=8712,
        #                 srate=1000, chanlocs=['POz', 'PO3', 'PO4', 'PO5', 'PO6', 'Oz', 'O1', 'O2', 'TRG'], n_chan=9)
        neuracle = dict(device_name='Neuracle', hostname='127.0.0.1', port=8712,
                        srate=250, chanlocs=['Fpz', 'Fp1', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8', 'Fz', 'F1', 'F2', 'F3', 'F4', 'F5',
                                              'F6', 'F7', 'F8', 'FCz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8',
                                               'CP5', 'CP6', 'TP7', 'TP8', 'Pz', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'POz', 'PO3',
                                              'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'Oz', 'O1', 'O2', 'ECG', 'HEOR', 'HEOL', 'VEOU',
                                              'VEOL','TRG'], n_chan=65)
        device = neuracle
        ### 选着设备型号,默认Neuracle
        self.target_device = device[0]
        ## 初始化 DataServerThread 线程
        time_buffer = 3  # second
        self.thread_data_server = DataServerThread(device=self.target_device['device_name'], n_chan=self.target_device['n_chan'],
                                              srate=self.target_device['srate'], t_buffer=time_buffer)
        ### 建立TCP/IP连接
        notconnect = self.thread_data_server.connect(hostname=self.target_device['hostname'], port=self.target_device['port'])
        if notconnect:
            raise TypeError("Can't connect recorder, Please open the hostport ")
        else:
            # 启动线程
            self.thread_data_server.Daemon = True
            self.thread_data_server.start()
            print('Data server connected')

    def guidance(self):
        self.clean_screen()
        self.wait(2)
        self.show_ml_text('接下来你需要按照要求完成一些任务', (270, 210))
        self.wait_space()
        self.clean_screen()
        self.show_ml_text('出现“+”时集中精力\n先观看图像1秒\n后开始想象图像3秒\n尽量减少眨眼以及其他动作', (270, 210))
        self.wait_space()
        self.clean_screen()

    def run_task(self, task):
        code = self.code_book[task]
        self.draw(self.cross)
        self.wait(1)
        self.clean_screen()
        self.draw(self.mosaic)
        self.wait(0.5)
        self.clean_screen()
        self.show_ml_text(task, (1200, 100))
        pg.draw.rect(self.rect, self.font_color, self.rect.get_rect(), 10)
        self.draw(self.rect)
        self.draw(self.images[self.tasks.index(task)])
        self.wait(1)
        self.clean_screen()
        self.stimulate_code = self.trigger(int(code))
        self.logger('{},{},图片刺激'.format(self.stimulate_code, task))
        self.draw(self.mosaic)
        self.wait(0.5)
        self.clean_screen()
        pg.draw.rect(self.rect, self.font_color, self.rect.get_rect(), 10)
        self.draw(self.rect)
        self.wait(3)
        self.clean_screen()
        imagine_code = self.trigger(int(code + 1))
        self.logger('{},{},想象图片'.format(imagine_code, task))

    def main_body(self):
        if self.type == 1: #有间隔刺激
            for i in range(self.num_per_event):
                self.show_ml_text('按空格开始', (270, 210))
                self.wait_space()
                self.clean_screen()
                for t in self.tasks:
                    self.run_task(t)
        else: #单张图重复刺激
            for t in self.tasks:
                self.show_ml_text('按空格开始', (270, 210))
                self.wait_space()
                self.clean_screen()
                for i in range(self.num_per_event):
                    self.run_task(t)

    def conclusion(self):
        self.wait(1)
        self.show_text_center('实验结束')
        self.wait(1)

    def main(self):
        self.guidance()
        self.get_data_thread()
        self.main_body()
        self.conclusion()
        ## 结束线程
        self.thread_data_server.stop()
        self.terminate()

    def trigger(self, code):
        # triggerbox = TriggerBox("COM3")
        print('send trigger: {}'.format(code))
        # triggerbox.output_event_data(code)
        return code


    def get_data_thread(self):
        my_thread = threading.Thread(target=self.get_data)
        # 启动新线程
        my_thread.start()
    def get_data(self):
        while True:
            time.sleep(1)
            nUpdate =self.thread_data_server.GetDataLenCount()
            if nUpdate > (1 * self.target_device['srate'] - 1):
                data = self.thread_data_server.GetBufferData()
                self.thread_data_server.ResetDataLenCount()
                triggerChan = data[-1, :]
                idx = np.argwhere(triggerChan > 0)
                print(idx)
                # print(data.shape)
                # print(data.dtype)
                save_matrix_to_csv(np.transpose(data), f'csv_data/raw20230903_{13}.csv')
                print(data.shape)
        # pass


if __name__ == '__main__':

    baton = pg.image.load('image/task3/baton.jpg')
    boy = pg.image.load('image/task3/boy.jpg')
    domino = pg.image.load('image/task3/domino.jpg')
    eyeliner = pg.image.load('image/task3/eyeliner.jpg')
    honeycomb = pg.image.load('image/task3/honeycomb.jpg')
    limousine = pg.image.load('image/task3/limousine.jpg')
    mulch = pg.image.load('image/task3/mulch.jpg')
    shopping_cart = pg.image.load('image/task3/shopping_cart.jpg')
    skin = pg.image.load('image/task3/skin.jpg')
    torso = pg.image.load('image/task3/torso.jpg')

    width = 1200
    height = 900
    baton = pg.transform.scale(baton, (width, height))
    boy = pg.transform.scale(boy, (width, height))
    domino = pg.transform.scale(domino, (width, height))
    eyeliner = pg.transform.scale(eyeliner, (width, height))
    honeycomb = pg.transform.scale(honeycomb, (width, height))
    limousine = pg.transform.scale(limousine, (width, height))
    mulch = pg.transform.scale(mulch, (width, height))
    shopping_cart = pg.transform.scale(shopping_cart, (width, height))
    skin = pg.transform.scale(skin, (width, height))
    torso = pg.transform.scale(torso, (width, height))

    # images1 = [bonnet, cash_machine, coal, diamond, flower, headphones, hovercraft, hummingbird, lego, moss]
    # tasks1 = ['帽子', '自动取款机', '煤炭', '钻石', '花朵', '耳机', '气垫船', '蜂鸟', '乐高积木', '苔藓']
    #
    # images2 = [bonnet2, credit_card, diskette, goose, hovercraft2, ice_pack, letter_opener, locker, seahorse, sorbet]
    # tasks2 = ['帽子', '信用卡', '磁盘', '鹅', '气垫船', '冰袋', '开信器', '柜子', '海马', '果汁冰糕']

    images3 = [baton, boy, domino, eyeliner, honeycomb, limousine, mulch, shopping_cart, skin, torso]
    tasks3 = ['指挥棒', '男孩', '多米诺卡牌', '眼线笔', '蜂巢', '豪华轿车', '护根', '购物车', '皮肤', '躯干']

    task = MotionImg(images3, tasks3, type=1, num_per_event=2)
    task.main()
