import cv2
import PIL
from PIL import Image
from typing import List


class Augmentator:
    def augment_images(self, images: List[Image]):
        result = []

        for image in images:
            for angle in [0, 90, 180, 270]:
                rot = image.rotate(angle)
                result.append(rot)
                result.append(rot.transpose(PIL.Image.FLIP_LEFT_RIGHT))
                result.append(rot.transpose(PIL.Image.FLIP_TOP_BOTTOM))

        return result

    def rotate_image(self, input_image, axis):
        return cv2.flip(input_image, axis)
