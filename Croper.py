import os

import numpy as np
import datetime
import keyboard

from mss import mss

sct = mss()

import dxcam

camera = dxcam.create()


def crop_with_size(img, size):
    w = img.width
    h = img.height
    x0 = (w - size) / 2
    y0 = (h - size) / 2
    x1 = (w + size) / 2
    y1 = (h + size) / 2
    return img.crop((x0, y0, x1, y1))


def crop_with_size_array(np_arr, size):
    w = np_arr.shape[1]
    h = np_arr.shape[0]
    x0 = int((w - size) / 2)
    y0 = int((h - size) / 2)
    x1 = int((w + size) / 2)
    y1 = int((h + size) / 2)
    return np_arr[y0:y1, x0:x1]


def get_best_crop_size(screen_size, screen_crop_factor=25, is_round_pow2=False):
    best = np.sqrt(screen_size / screen_crop_factor)
    if is_round_pow2:
        return 2 ** round(np.log2(best))
    return best


class Croper:
    def __init__(self):

        screen = camera.grab()

        self.screen_width = screen.shape[1]
        self.screen_height = screen.shape[0]

        self.images_folder = "images"
        self.crop_size = 512

        x0 = (self.screen_width - self.crop_size) // 2
        y0 = (self.screen_height - self.crop_size) // 2
        x1 = (self.screen_width + self.crop_size) // 2
        y1 = (self.screen_height + self.crop_size) // 2

        self.bounding_box = {'top': y0, 'left': x0, 'width': self.crop_size, 'height': self.crop_size}

        self.region = (x0, y0, x1, y1)

    def set_crop_bounding_box(self):
        x0 = (self.screen_width - self.crop_size) // 2
        y0 = (self.screen_height - self.crop_size) // 2
        x1 = (self.screen_width + self.crop_size) // 2
        y1 = (self.screen_height + self.crop_size) // 2

        self.region = (x0, y0, x1, y1)

    def crop(self, img):
        return crop_with_size(img, self.crop_size)

    def crop_screen_array(self):
        return camera.grab(region=self.region)

    def crop_screen_array_mss(self):
        return np.array(sct.grab(self.bounding_box))[..., [2, 1, 0]]

    def crop_loop_screen(self):
        while True:
            if keyboard.is_pressed('x'):
                croped = self.crop_screen()
                time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                croped_path = os.path.join(self.images_folder, f"crop_{time}.png")
                croped.save(croped_path)
