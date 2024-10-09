import datetime
import os

import numpy as np
import keyboard
import pygame as pg

from Croper import Croper, crop_with_size_array
from Recognizer import Recognizer, show_images_row
import mouse

import asyncio

pg.mixer.init()

program_start_sound = pg.mixer.Sound(os.path.join("sounds", "program_start.mp3"))
screen_shot_sound = pg.mixer.Sound(os.path.join("sounds", "screen_shot.mp3"))
on_sound = pg.mixer.Sound(os.path.join("sounds", "ON.wav"))
off_sound = pg.mixer.Sound(os.path.join("sounds", "OFF.wav"))
terrorist_sound = pg.mixer.Sound(os.path.join("sounds", "terrorist.mp3"))
counter_terrorists_sound = pg.mixer.Sound(os.path.join("sounds", "counter_terrorist.mp3"))


class Autoshot:
    def __init__(self):
        self.key_press_delay = 0.7
        self.predict_delay = 0.0

        self.croper = Croper()
        self.croper.crop_size = 512  # self.get_best_crop_size(2560 * 1440)
        self.croper.set_crop_bounding_box()

        self.recognizer = Recognizer()
        self.recognizer.load_model("save50.keras")
        self.auto_check = False
        self.show_predictions = False

        # 0 - None, 1 - CT, 2 - T
        self.enemy_class = 1

    def is_need_shoot(self):

        croped = self.croper.crop_screen_array_mss()

        pred = self.recognizer.predict(croped)

        center = crop_with_size_array(pred, 10)
        obj_class = np.median(center)
        proba = np.count_nonzero(center == obj_class) / center.size

        if self.show_predictions:
            pred_image = self.recognizer.mask_array_to_image_with_background(pred, croped)
            show_images_row([croped, pred_image])
        if obj_class == self.enemy_class and proba >= 0.5:
            return True

        return False

    def loop(self):
        print("Loop started!")
        program_start_sound.play()

        key_time_last = datetime.datetime.now()
        predict_time_last = datetime.datetime.now()

        while True:
            predict_time_now = key_time_now = datetime.datetime.now()

            delta_key = (key_time_now - key_time_last).total_seconds()
            can_press_key = delta_key >= self.key_press_delay

            delta_predict = (predict_time_now - predict_time_last).total_seconds()
            can_predict = delta_predict >= self.predict_delay

            if keyboard.is_pressed('F1') and can_press_key:
                key_time_last = key_time_now
                screen_shot_sound.play()

                croped = self.croper.crop_screen()
                time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                croped_path = os.path.join(self.croper.images_folder, f"crop_{time}.png")
                croped.save(croped_path)
            elif keyboard.is_pressed('F2') and can_press_key:
                key_time_last = key_time_now
                self.auto_check = not self.auto_check
                if self.auto_check:
                    print("Auto shot ON")
                    on_sound.play()
                else:
                    print("Auto shot OFF")
                    off_sound.play()
            elif keyboard.is_pressed('F3') and can_press_key:
                key_time_last = key_time_now
                if self.enemy_class == 1:
                    print("Enemy - T")
                    self.enemy_class = 2
                    terrorist_sound.play()
                else:
                    print("Enemy - CT")
                    self.enemy_class = 1
                    counter_terrorists_sound.play()
            if can_predict and (self.auto_check or keyboard.is_pressed('F4')):
                is_need_shoot = self.is_need_shoot()
                predict_time_last = predict_time_now
                if is_need_shoot:
                    mouse.click("left")
            if keyboard.is_pressed('F5') and can_press_key:
                key_time_last = key_time_now
                self.show_predictions = not self.show_predictions

                if self.show_predictions:
                    print("Show predictions ON")
                    on_sound.play()
                else:
                    print("Show predictions OFF")
                    off_sound.play()


autoshot = Autoshot()

autoshot.loop()