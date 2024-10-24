# -*- coding: utf-8 -*-
import random
import os
import numpy as np
import pygame as pg
import time
from utils.task import Task

from Jellyfish_Python_API.neuracle_api import DataServerThread
from neuracle_lib.triggerBox import TriggerBox, PackageSensorPara

import csv


def save_matrix_to_csv(matrix, filename):
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(matrix)


class MotionImg(Task):
    def __init__(self, images=None, tasks=None, num_per_event=10):
        super().__init__(exp_name='image')
        self.button_width = 300
        self.button_height = 300
        self.button_spacing = 40
        self.image_width = 700
        self.image_height = 700
        self.mosaic = pg.image.load('image/noise.jpg')
        self.mosaic = pg.transform.scale(self.mosaic, (self.image_width, self.image_height))
        self.bg_color = (0, 0, 0)
        self.font_color = (255, 255, 255)
        self.num_per_event = num_per_event
        self.tasks = tasks
        self.code_book = dict(zip(self.tasks, np.arange(50, 70 + 20 * len(self.tasks), 20)))
        self.cross = pg.font.Font(self.default_font, self.resize_value(200)).render('+', True, 'white')
        self.rect = pg.Surface((self.image_width+20, self.image_height+20))
        self.rect.fill(self.bg_color)
        self.original_images = images
        self.task_images = [row[0] for row in self.original_images]
        self.task_images = self.resize_image(self.task_images, self.image_width, self.image_height)
        self.image_buttons = [self.resize_image(x, self.button_width, self.button_height) for x in self.original_images]
        self.task_buttons = [row[0] for row in self.image_buttons]
        self.indexed_images = list(enumerate(self.task_buttons))
        self.logger(str(self.code_book))
        self.logger('code, image')

        self.buttons = self.create_buttons()
        self.answers = []
        self.results = []

        self.sample_rate = 1000
        self.t_buffer = 10000
        self.thread_data_server = DataServerThread(self.sample_rate, self.t_buffer)
        self.flagstop = False
        self.triggerbox = TriggerBox("COM4")

    def main(self):
        self.guidance()
        self.start_server()
        self.main_body()
        self.conclusion()
        self.save_data()
        self.terminate()

    def guidance(self):
        self.clean_screen()
        self.wait(2)
        self.show_ml_text('接下来你需要按照要求完成一些任务', (20, 20))
        self.wait_space()
        self.clean_screen()

    def main_body(self):
        self.exp1()
        self.exp2()
        self.exp3()

    def conclusion(self):
        self.wait(2)
        self.show_text_center('实验结束')
        self.wait(2)
        self.clean_screen()

    def terminate(self, code: int = 0):
        self.logger('Exit time: ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        self.flagstop = True
        self.thread_data_server.stop()
        pg.quit()
        exit(code)

    def start_server(self):
        notconnect = self.thread_data_server.connect(hostname='127.0.0.1', port=8712)
        if notconnect:
            raise TypeError("Can't connect JellyFish, Please open the hostport ")
        else:
            # meta包还没解析好就等待
            while not self.thread_data_server.isReady():
                time.sleep(1)
                continue
            # 启动线程
            self.thread_data_server.start()
            print('Data server start')

    def save_data(self):
        self.flagstop = True
        self.thread_data_server.stop()
        data = self.thread_data_server.GetBufferData()
        self.show_text_center('正在保存数据，请等待...')
        np.save(f'npy_data/20231020-qyz-{1}.npy', data)
        self.clean_screen()
        self.show_text_center('数据保存完成')
        self.wait(2)

    def create_buttons(self):
        screen_width = self.screen_size[0]
        screen_height = self.screen_size[1]
        buttons = []
        button_width = self.button_width
        button_height = self.button_height
        button_spacing = self.button_spacing
        for i in range(10):
            if i < 5:
                button_x = (screen_width - (button_width + button_spacing) * 5) // 2 + i * (
                            button_width + button_spacing)
                button_y = (screen_height - (button_height + button_spacing) * 2) // 2
            else:
                button_x = (screen_width - (button_width + button_spacing) * 5) // 2 + (i - 5) * (
                            button_width + button_spacing)
                button_y = (screen_height - (button_height + button_spacing) * 2) // 2 + button_height + button_spacing
            button_rect = pg.Rect(button_x, button_y, button_width, button_height)
            buttons.append(button_rect)
        return buttons

    def resize_image(self, images_list, image_width, image_height):
        image_buttons = []
        for image in images_list:
            image = pg.transform.scale(image, (image_width, image_height))
            image_buttons.append(image)
        return image_buttons

    def trigger(self, code):
        self.triggerbox.output_event_data(code)
        return code

    def exp1(self):
        self.clean_screen()
        self.wait(2)
        self.show_ml_text('实验1：记忆实验\n\n'
                          '实验说明：\n'
                          '该实验分为4个子实验a,b,c,d\n'
                          '被试需要在子实验a中记住10张图片及其所对应的概念\n'
                          '在子实验b, c, d中进行测试，测试通过则进行实验2，否则重复实验1', (20, 20))
        self.wait_space()
        self.clean_screen()
        count = 0
        while count < 3:
            count += 1
            self.learning()
            # self.test1()
            # self.test2()
            self.test3()
            accuracy = self.accuracy_calculating()
            print(f'accuracy: {accuracy}')
            if accuracy >= 0.9:
                self.show_ml_text(f'你的准确率为：{accuracy*100}%\n'
                                  f'通过记忆测试，按空格开始实验2', (20, 20))
                self.wait_space()
                self.clean_screen()
                break
            else:
                self.show_ml_text(f'你的准确率为：{accuracy*100}%\n'
                                  f'没通过记忆测试，按空格重复实验1', (20, 20))
                self.wait_space()
                self.clean_screen()

    def learning(self):
        self.show_ml_text('子实验a：\n'
                          '出现“+”时集中精力\n'
                          '屏幕中会随机展示一张图片及其所对应的概念3秒\n'
                          '被试尽力记住这些图片', (20, 20))
        self.wait_space()
        self.clean_screen()
        text_position = self.get_center_position((self.image_width, self.image_height))
        for i in range(5):
            indexed_tasks = list(enumerate(self.tasks))
            random.shuffle(indexed_tasks)
            for current_index, (original_index, task) in enumerate(indexed_tasks):
                self.draw(self.cross)
                self.wait(1)
                self.clean_screen()
                self.draw(self.task_images[self.tasks.index(task)])
                self.show_ml_text(f'{task}', (text_position[0], text_position[1]-90))
                self.wait(1)
                self.clean_screen()

    def test1(self):
        self.show_ml_text('子实验b：\n'
                          '根据图片选择概念', (270, 210))
        self.wait_space()
        self.clean_screen()
        indexed_tasks = list(enumerate(self.tasks))
        random.shuffle(indexed_tasks)
        for current_index, (original_index, task) in enumerate(indexed_tasks):
            self.answers.append(original_index)
            self.draw(self.cross)
            self.wait(1)
            self.clean_screen()
            self.draw(self.task_images[self.tasks.index(task)])
            self.wait(1)
            self.clean_screen()
            self.wait_left(dur=3)
            self.clean_screen()
            self.show_result(self.answers[-1], self.results[-1], task)

    def test2(self):
        self.show_ml_text('子实验c：\n'
                          '根据概念选择图片', (20, 20))
        self.wait_space()
        self.clean_screen()
        indexed_tasks = list(enumerate(self.tasks))
        random.shuffle(indexed_tasks)
        random.shuffle(self.indexed_images)
        for current_index, (original_index, task) in enumerate(indexed_tasks):
            self.answers.append(self.indexed_images.index((original_index, self.task_buttons[original_index])))
            self.draw(self.cross)
            self.wait(1)
            self.clean_screen()
            self.show_text_center(task)
            self.wait(1)
            self.clean_screen()
            self.wait_left(dur=3, images=[x[1] for x in self.indexed_images])
            self.clean_screen()
            self.show_result(self.answers[-1], self.results[-1], task)

    def test3(self):
        self.show_ml_text('子实验d：\n'
                          '在同类的十张图片中选择特定的图片', (270, 210))
        self.wait_space()
        self.clean_screen()
        indexed_tasks = list(enumerate(self.tasks))
        random.shuffle(indexed_tasks)
        for current_index, (original_index, task) in enumerate(indexed_tasks):
            class_images = self.image_buttons[original_index]
            indexed_image = list(enumerate(class_images))
            random.shuffle(indexed_image)
            self.answers.append(indexed_image.index((0, class_images[0])))
            self.show_text_center(task)
            self.wait(1)
            self.clean_screen()
            self.wait_left(dur=3, images=[x[1] for x in indexed_image])
            self.clean_screen()
            self.show_result(self.answers[-1], self.results[-1], task)

    def show_result(self, answer, result, task):
        if answer != result:
            self.draw(self.task_images[self.tasks.index(task)])
            if self.results[-1] == -1:
                self.show_ml_text(f'你已超时！\n'
                                  f'正确答案：{task}\n'
                                  f'点击鼠标右键继续', (20, 20))
            else:
                self.show_ml_text(f'答案错误！\n'
                                  f'正确答案：{task}\n'
                                  f'点击鼠标右键继续', (20, 20))
            self.wait_right()
            self.clean_screen()

    def exp2(self):
        font = pg.font.Font("./font/Deng.ttf", 200)
        self.show_ml_text('实验2：\n'
                          '每轮开始前将采集3秒静息态脑电\n'
                          '出现“+”时集中精力\n'
                          '屏幕中会随机展示一个概念1秒\n'
                          '被试根据概念想象相应的图片3秒\n'
                          '总共进行10轮实验，大概需要10分钟\n'
                          '想象过程中尽量保持不动以及不要眨眼', (20, 20))
        self.wait_space()
        self.clean_screen()
        indexed_tasks = list(enumerate(self.tasks))
        random.shuffle(indexed_tasks)
        relax_trigger = 1
        start_trigger = 2
        for i in range(self.num_per_event):
            self.show_text_center('采集静息态脑电，保持不动')
            self.wait(1)
            self.clean_screen()
            self.trigger(relax_trigger)
            self.wait(3)
            for current_index, (original_index, task) in enumerate(indexed_tasks):
                code = self.code_book[task]
                text = font.render(task, True, self.font_color)
                self.draw(self.cross)
                self.wait(1)
                self.clean_screen()
                self.draw(self.mosaic)
                self.wait(0.5)
                self.clean_screen()
                self.draw(text)
                self.wait(1)
                self.clean_screen()
                self.draw(self.mosaic)
                self.wait(0.5)
                self.clean_screen()
                pg.draw.rect(self.rect, self.font_color, self.rect.get_rect(), 10)
                self.draw(self.rect)
                self.trigger(start_trigger)
                self.wait(3)
                self.trigger(code)
                self.logger('{},{},被动想象'.format(code, task))
                self.clean_screen()
            self.show_ml_text(f'第{i + 1}轮结束，共10轮，按空格继续', (20, 20))
            self.wait_space()
            self.clean_screen()

    def review(self):
        self.show_text_center('先回顾一次图片')
        self.wait(1)
        self.clean_screen()
        text_position = self.get_center_position((self.image_width, self.image_height))
        indexed_tasks = list(enumerate(self.tasks))
        random.shuffle(indexed_tasks)
        for current_index, (original_index, task) in enumerate(indexed_tasks):
            self.draw(self.cross)
            self.wait(1)
            self.clean_screen()
            self.draw(self.task_images[self.tasks.index(task)])
            self.show_ml_text(f'{task}', (text_position[0], text_position[1] - 90))
            self.wait(1)
            self.clean_screen()

    def exp3(self):
        self.show_ml_text('实验3：\n'
                          '被试主动想象图片3秒\n'
                          '然后选择其概念\n'
                          '总共进行100次主动想象任务，大概需要7分钟\n'
                          '想象过程中尽量保持不动以及不要眨眼', (20, 20))
        self.wait_space()
        self.clean_screen()
        self.review()
        self.show_text_center('按空格开始实验3')
        self.wait_space()
        self.clean_screen()
        start_trigger = 3
        for i in range(self.num_per_event):
            for j in range(10):
                self.show_text_center('点击鼠标右键进行想象')
                self.wait_right()
                self.clean_screen()
                self.draw(self.mosaic)
                self.wait(0.5)
                pg.draw.rect(self.rect, self.font_color, self.rect.get_rect(), 10)
                self.draw(self.rect)
                self.trigger(start_trigger)
                self.wait(3)
                self.wait_left(type=1)
                self.clean_screen()
            # self.show_ml_text(f'第{i + 1}轮结束，按空格继续', (20, 20))
            # self.wait_space()

    def draw_image_buttons(self, images):
        screen_width = self.screen_size[0]
        screen_height = self.screen_size[1]
        button_width = self.button_width
        button_height = self.button_height
        button_spacing = self.button_spacing
        for i in range(10):
            if i < 5:
                button_x = (screen_width - (button_width + button_spacing) * 5) // 2 + i * (
                        button_width + button_spacing)
                button_y = (screen_height - (button_height + button_spacing) * 2) // 2
            else:
                button_x = (screen_width - (button_width + button_spacing) * 5) // 2 + (i - 5) * (
                        button_width + button_spacing)
                button_y = (screen_height - (button_height + button_spacing) * 2) // 2 + button_height + button_spacing
            self.screen.blit(images[i], (button_x, button_y))

    def draw_buttons(self):
        font = pg.font.Font("./font/Deng.ttf", 60)
        for i, button in enumerate(self.buttons):
            pg.draw.rect(self.screen, self.font_color, button)
            text = font.render(self.tasks[i], True, self.bg_color)
            text_rect = text.get_rect(center=button.center)
            self.screen.blit(text, text_rect)

    def click_solver(self, type: int = 0):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.terminate(1)
            elif event.type == pg.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键点击
                    for i, button in enumerate(self.buttons):
                        if button.collidepoint(event.pos):
                            if type == 0:
                                self.results.append(i)
                            else:
                                code = self.code_book[self.tasks[i]]
                                self.trigger(code)
                                self.logger('{},{},主动想象'.format(code, self.tasks[i]))
                                self.clean_screen()
                            return True
        return False

    def wait_left(self, dur: float = -1, images: list = None, type: int = 0) -> bool:
        pg.event.clear()
        limit = time.perf_counter() + dur
        while (dur == -1) or (limit > time.perf_counter()):
            self.screen.fill(self.bg_color)
            if images is None:
                self.draw_buttons()
            else:
                self.draw_image_buttons(images)
            mouse_pos = pg.mouse.get_pos()
            pg.draw.circle(self.screen, (255, 0, 0), mouse_pos, 5)
            pg.display.flip()
            if self.click_solver(type):
                return True
        print("time out")
        self.results.append(-1)
        return False

    def wait_right(self, dur: float = -1):
        pg.event.clear()
        limit = time.perf_counter() + dur
        while (dur == -1) or (limit > time.perf_counter()):
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.terminate(1)
                elif event.type == pg.MOUSEBUTTONDOWN and event.button == 3:  # 鼠标右键点击
                    return True
        return False

    def accuracy_calculating(self):
        list_length = len(self.answers)
        correct_count = sum(1 for x, y in zip(self.answers, self.results) if x == y)
        accuracy = correct_count/list_length
        return accuracy


if __name__ == '__main__':
    tasks = ['指挥棒', '男孩', '多米诺卡牌', '眼线笔', '蜂巢', '豪华轿车', '护根', '购物车', '皮肤', '躯干']
    classes = ['baton', 'boy', 'domino', 'eyeliner', 'honeycomb', 'limousine', 'mulch', 'shopping_cart', 'skin',
               'torso']
    images = []
    root_dir = 'image'
    for class_name in classes:
        class_list = []
        class_path = os.path.join(root_dir, class_name)
        print(class_path)
        for i in range(1):
            file_name = f'{class_name}_{i}.jpg'
            file_path = os.path.join(class_path, file_name)
            image = pg.image.load(file_path)
            class_list.append(image)
        images.append(class_list)

    task = MotionImg(images, tasks, num_per_event=1)
    task.main()
