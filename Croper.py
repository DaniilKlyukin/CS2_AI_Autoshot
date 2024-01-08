import os

import numpy as np
from PIL import ImageGrab
import datetime
import keyboard


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
        self.images_folder = "images"
        self.crop_size = 512
        self.crop_resize = 512

    def crop(self, img):
        return crop_with_size(img, self.crop_size).resize((self.crop_resize, self.crop_resize))

    def crop_screen(self):
        img = ImageGrab.grab()
        return self.crop(img)

    def crop_loop_screen(self):
        while True:
            if keyboard.is_pressed('x'):
                croped = self.crop_screen()
                time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                croped_path = os.path.join(self.images_folder, f"crop_{time}.png")
                croped.save(croped_path)
