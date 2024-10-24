# -*- coding: utf-8 -*-
import random
import os
import numpy as np
import pygame as pg
import time
# from utils.task import TaskModel

from Jellyfish_Python_API.neuracle_api import DataServerThread
from neuracle_lib.triggerBox import TriggerBox, PackageSensorPara

class TaskModel:
    def __init__(self, images, imageNames, tasks, type, num_per_event):
        self.images = images
        self.imageNames = imageNames
        self.tasks = tasks
        self.type = type
        self.num_per_event = num_per_event
        self.current_task_index = 0
        self.current_phase = 'guidance'
        self.current_sequence = 0
        self.total_sequences = len(images) // 20
        self.sample_rate = 1000
        self.t_buffer = 1000
        self.thread_data_server = DataServerThread(self.sample_rate, self.t_buffer)
        self.flagstop = False
        self.triggerbox = TriggerBox("COM4")
        # 使用任务名称（即子目录名称）作为触发器代码
        self.images_with_labels = [(image, (task.split('_')[0]+imageName[-7:-5])) for image, task, imageName in zip(images, tasks, imageNames)]

        self.sequence_indices = list(range(len(images)))  # 创建一个索引列表
        print("图像数量：", len(images))
        random.shuffle(self.sequence_indices)  # 打乱索引列表

    def trigger(self, label):
        code = int(label)  # 直接将传入的类别编号转换为整数
        print(f'Sending trigger for label {label}: {code}')
        self.triggerbox.output_event_data(code)

    def start_data_collection(self):
        notconnect = self.thread_data_server.connect(hostname='127.0.0.1', port=8712)
        if notconnect:
            raise Exception("Can't connect to JellyFish, please check the hostport.")
        else:
            while not self.thread_data_server.isReady():
                time.sleep(1)
                continue
            self.thread_data_server.start()

    def stop_data_collection(self):
        self.flagstop = True
        self.thread_data_server.stop()

    def save_data(self):
        data = self.thread_data_server.GetBufferData()
        np.save(f'yiming/{time.strftime("%Y%m%d-%H%M%S")}-data.npy', data)
        print("Data saved!")

    def get_next_sequence(self):
        # 确保不会超出列表范围
        if self.current_sequence * self.num_per_event >= len(self.sequence_indices):
            raise Exception("All sequences have been displayed.")

        # 从打乱的索引列表中获取下一个序列的索引
        start_index = self.current_sequence * self.num_per_event
        end_index = start_index + self.num_per_event
        sequence_indices = self.sequence_indices[start_index:end_index]

        # 更新当前序列计数
        self.current_sequence += 1

        # 返回选中的图像和标签，即返回 20 个 images_with_labels 元组
        return [self.images_with_labels[i] for i in sequence_indices]

    def reset_sequence(self):
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
        pg.draw.circle(self.screen, (255, 0, 0), (600, 450), 30, 0)
        # 绘制黑色十字
        pg.draw.line(self.screen, (0, 0, 0), (575, 450), (625, 450), 10)
        pg.draw.line(self.screen, (0, 0, 0), (600, 425), (600, 475), 10)
        pg.display.flip()

    def display_image(self, image):
        self.screen.blit(image, (0, 0))
        pg.draw.circle(self.screen, (255, 0, 0), (600, 450), 30, 0)
        pg.draw.line(self.screen, (0, 0, 0), (575, 450), (625, 450), 10)
        pg.draw.line(self.screen, (0, 0, 0), (600, 425), (600, 475), 10)

        # 更新屏幕显示
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
        self.model.start_data_collection()
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        running = False
                    elif event.key == pg.K_SPACE:
                        if self.model.current_phase == 'guidance_waiting':
                            self.model.set_phase('black_screen_pre')
                        elif self.model.current_phase == 'blink_time':
                            self.model.set_phase('black_screen_pre')

            # 实验指导阶段
            if self.model.current_phase == 'guidance':
                self.view.display_multiline_text(
                    '接下来你需要按照要求完成一些任务:\n出现“+”时集中精力\n开始观看图像\n尽量减少眨眼以及其他动作',
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
                image_label_sequence = self.model.get_next_sequence()
                for image_label_pair in image_label_sequence:
                    image, label = image_label_pair
                    print("label: ", label)
                    self.view.display_image(image)
                    self.model.trigger(label)  # 使用图像的类别编号发送触发器
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
                time.sleep(3)  # 展示5秒结束文字
                running = False

        # 在实验循环结束后停止数据收集并保存数据
        self.model.stop_data_collection()
        self.model.save_data()  # 保存数据
        pg.quit()

    def handle_space(self):
        # 按空格键跳转到下一个序列的开始
        if self.model.current_phase == 'blink_time':
            self.model.set_phase('black_screen_pre')


if __name__ == '__main__':
    base_dir = r"C:\Users\ncclab\PycharmProjects\CognitiveTaskSet\training_images"
    images = []
    imageNames = []
    tasks = []
    max_folders = 10
    folder_count = 0

    for subdir in sorted(os.listdir(base_dir)):
        print("读取目录：", subdir)
        if folder_count >= max_folders:
            break

        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            for file in sorted(os.listdir(subdir_path)):
                if file.endswith((".jpg", ".png")):
                    image_path = os.path.join(subdir_path, file)
                    image = pg.image.load(image_path)
                    image = pg.transform.scale(image, (1200, 900))
                    images.append(image)
                    imageNames.append(file)
                    tasks.append(subdir)
            folder_count += 1

    model = TaskModel(images, imageNames, tasks, type=1, num_per_event=20)
    view = TaskView()
    controller = TaskController(model, view)
    controller.run()
