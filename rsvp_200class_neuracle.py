import numpy as np
import pygame as pg

from utils.task import Task
import sys
import os
import csv
import threading
import random
import pygame as pg
import time
import argparse
from neuracle_lib.triggerBox import TriggerBox,PackageSensorPara
from Jellyfish_Python_API.neuracle_api import DataServerThread
from neuracle_lib.dataServer import DataServerThread
import time


class TaskModel:
    def __init__(self, images, tasks, type, num_per_event):
        self.images = images
        self.tasks = tasks
        self.type = type
        self.num_per_event = num_per_event
        self.current_task_index = 0
        self.current_phase = 'guidance'
        self.current_sequence = 0  # 当前序列号
        self.total_sequences = len(images) // 20  # 计算总共有多少个序列

    def get_next_sequence(self):
        # 在获取序列前先打乱整个图片列表
        random.shuffle(self.images)
        # 获取下一序列的图片
        start_index = self.current_sequence * 20
        end_index = start_index + 20
        self.current_sequence += 1
        return self.images[start_index:end_index]

    def reset_sequence(self):
        # 重置序列号
        self.current_sequence = 0
    def set_phase(self, phase):
        self.current_phase = phase

class TaskView:
    def __init__(self):
        pg.init()
        self.screen = pg.display.set_mode((1200, 900))
        pg.display.set_caption('Task')
        # 指定一个支持中文的字体文件
        self.font_path = "C:/Windows/Fonts/msyhbd.ttc"

    def display_fixation(self):
        self.screen.fill((0, 0, 0))  # 清屏
        # 绘制红色圆
        pg.draw.circle(self.screen, (255, 0, 0), (600, 450), 50, 0)
        # 绘制黑色十字
        pg.draw.line(self.screen, (0, 0, 0), (575, 450), (625, 450), 10)
        pg.draw.line(self.screen, (0, 0, 0), (600, 425), (600, 475), 10)
        pg.display.flip()

    def display_image(self, image):
        self.screen.blit(image, (0, 0))
        pg.display.flip()

    def display_text(self, text, position):
        # 使用指定的中文字体渲染文本
        font = pg.font.Font(self.font_path, 50)
        text_surface = font.render(text, True, (255, 255, 255))
        self.screen.blit(text_surface, position)
        pg.display.flip()

    def clear_screen(self):
        self.screen.fill((0, 0, 0))
        pg.display.flip()

    def display_multiline_text(self, text, position, font_size, line_spacing):
        font = pg.font.Font(self.font_path, font_size)
        lines = text.splitlines()  # 分割文本为多行
        x, y = position

        for line in lines:
            line_surface = font.render(line, True, (255, 255, 255))
            self.screen.blit(line_surface, (x, y))
            y += line_surface.get_height() + line_spacing  # 更新y坐标，为下一行做准备

        pg.display.flip()  # 更新屏幕显示


class TaskController:
    def __init__(self, model, view):
        self.model = model
        self.view = view

    def run(self):
        running = True
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:  # 检测ESC键
                        running = False  # 设置标志以结束实验
                    elif event.key == pg.K_SPACE:
                        if self.model.current_phase == 'guidance_waiting':
                            self.model.set_phase('black_screen_pre')
                        elif self.model.current_phase == 'blink_time':
                            self.model.set_phase('black_screen_pre')

            # 实验指导阶段
            if self.model.current_phase == 'guidance':
                self.view.display_multiline_text(
                    '接下来你需要按照要求完成一些任务:\n出现“+”时集中精力\n先观看图像1秒\n后开始想象图像3秒\n尽量减少眨眼以及其他动作',
                    (50, 50), 50, 5)
                self.model.set_phase('guidance_waiting')
            # 实验指导等待阶段
            elif self.model.current_phase == 'guidance_waiting':
                # 等待空格键按下，无需额外操作，事件循环中已处理
                pass
            # 序列开始前的黑屏
            elif self.model.current_phase == 'black_screen_pre':
                self.view.clear_screen()
                time.sleep(0.75)  # 750ms 黑屏
                self.model.set_phase('show_images')

            # 修改展示图片序列的部分
            # 展示图片序列
            elif self.model.current_phase == 'show_images':
                images = self.model.get_next_sequence()  # 获取下一序列的图片
                for image in images:
                    self.view.display_fixation()
                    self.view.display_image(image)
                    time.sleep(0.1)
                    self.view.display_fixation()
                    time.sleep(0.1)
                if self.model.current_sequence >= self.model.total_sequences:
                    self.model.set_phase('conclusion')
                else:
                    self.model.set_phase('black_screen_post')

            # 眨眼时间等待按键继续
            elif self.model.current_phase == 'blink_time':
                self.view.display_text('请眨眼，准备好后按空格继续', (50, 50))
                self.waiting_for_space = True  # 开始等待空格键

            # 序列结束后的黑屏
            elif self.model.current_phase == 'black_screen_post':
                self.view.clear_screen()
                time.sleep(0.75)  # 750ms 黑屏
                self.model.set_phase('blink_time')

            # 眨眼时间
            elif self.model.current_phase == 'blink_time':
                self.view.display_text('请眨眼', (50, 50))
                time.sleep(2)  # 2秒眨眼时间
                self.model.set_phase('black_screen_pre')

            # 实验结束
            elif self.model.current_phase == 'conclusion':
                self.view.display_text('实验结束', (50, 50))
                time.sleep(5)  # 展示5秒结束文字
                running = False

        pg.quit()

    def handle_space(self):
        # 按空格键跳转到下一个序列的开始
        if self.model.current_phase == 'blink_time':
            self.model.set_phase('black_screen_pre')


if __name__ == '__main__':
    base_dir = "C:/Users/ncclab/PycharmProjects/CognitiveTaskSet/test"
    images = []
    tasks = []  # 如果需要，也可以填充任务名称

    # 遍历目录，加载图片
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            # 假设每个子目录下只有一张图片
            for file in os.listdir(subdir_path):
                if file.endswith((".jpg", ".png")):  # 根据实际情况可能需要调整图片格式
                    image_path = os.path.join(subdir_path, file)
                    image = pg.image.load(image_path)
                    image = pg.transform.scale(image, (1200, 900))  # 调整图片大小
                    images.append(image)
                    tasks.append(subdir)  # 可能需要根据您的需求调整

    model = TaskModel(images[:200], tasks[:200], type=1, num_per_event=2)
    view = TaskView()
    controller = TaskController(model, view)
    controller.run()