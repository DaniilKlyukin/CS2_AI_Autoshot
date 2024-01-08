import asyncio
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
from Augmentator import Augmentator
from tensorflow_examples.models.pix2pix import pix2pix
import keras


def load_images_folder(folder):
    images = []

    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            images.append(os.path.join(folder, filename))

    return images


def show_images_row(images):
    f, ax_arr = plt.subplots(1, len(images))
    for i, im in enumerate(images):
        ax_arr[i].imshow(im)
    plt.show()


def show_image_with_mask(image, mask):
    show_images_row([image, mask])


def show_images_with_masks(images, masks):
    for img, mask in zip(images, masks):
        show_image_with_mask(img, mask)


class Recognizer:
    def __init__(self):
        self.images_folder = "images"
        self.masks_folder = "masks"
        self.img_size = 512
        self.model = None
        self.save_dir_path = "saves"

        # mask - rgb
        # 0 - None, 1 - CT, 2 - T
        self.mask_map = [
            np.array([0, 0, 0]),
            np.array([0, 0, 255]),
            np.array([255, 0, 0])
        ]

    def load_images_with_masks(self):
        images_paths = load_images_folder(self.images_folder)
        masks_paths = load_images_folder(self.masks_folder)

        images = []
        masks = []

        for img_path, mask_path in zip(images_paths, masks_paths):
            img = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('RGB')

            images.append(img)
            masks.append(mask)

        return images, masks

    def normalize_image(self, img):
        image_arr = np.array(img)

        image_arr = tf.image.resize(image_arr, (self.img_size, self.img_size))

        image_arr = tf.cast(image_arr, tf.float32) / 255.0

        return image_arr

    def normalize_mask(self, img):
        image_arr = np.array(img)

        image_arr = tf.image.resize(image_arr, (self.img_size, self.img_size))

        r, g, b = image_arr[:, :, 0], image_arr[:, :, 1], image_arr[:, :, 2]

        delta_r = [r - map_rgb[0] for map_rgb in self.mask_map]
        delta_g = [g - map_rgb[1] for map_rgb in self.mask_map]
        delta_b = [b - map_rgb[2] for map_rgb in self.mask_map]

        dist = np.array([np.sqrt(dr ** 2 + dg ** 2 + db ** 2) for dr, dg, db in zip(delta_r, delta_g, delta_b)])

        amins = np.argmin(dist, axis=0)

        return amins

    def normalize_images(self, images):
        return [self.normalize_image(img) for img in images]

    def normalize_masks(self, images):
        return [self.normalize_mask(img) for img in images]

    def mask_array_to_image(self, mask):

        colors_array = [self.mask_map[val] for val in np.nditer(mask)]
        image = np.array(colors_array).reshape(mask.shape[0], mask.shape[1], 3)

        return image

    def mask_array_to_image_with_background(self, mask, background_image):

        background_array = np.array(background_image)
        colors_array = [self.mask_map[val] if val > 0 else background_array[i, j] for (i, j), val in
                        np.ndenumerate(mask)]
        image = np.array(colors_array).reshape(mask.shape[0], mask.shape[1], 3)

        return image

    def images_to_4d_array(self, images, channels):
        return np.array(images).reshape((len(images), self.img_size, self.img_size, channels))

    def init_model(self):

        base_model = tf.keras.applications.MobileNetV2(input_shape=[self.img_size, self.img_size, 3], include_top=False)

        # Use the activations of these layers
        layer_names = [
            'block_1_expand_relu',  # 64x64
            'block_3_expand_relu',  # 32x32
            'block_6_expand_relu',  # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',  # 4x4
        ]
        base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

        # Create the feature extraction model
        down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

        down_stack.trainable = False

        up_stack = [
            pix2pix.upsample(512, 3),  # 4x4 -> 8x8
            pix2pix.upsample(256, 3),  # 8x8 -> 16x16
            pix2pix.upsample(128, 3),  # 16x16 -> 32x32
            pix2pix.upsample(64, 3),  # 32x32 -> 64x64
        ]
        inputs = tf.keras.layers.Input(shape=[self.img_size, self.img_size, 3])

        # Downsampling through the model
        skips = down_stack(inputs)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        # This is the last layer of the model
        last = tf.keras.layers.Conv2DTranspose(
            filters=len(self.mask_map), kernel_size=3, strides=2,
            padding='same')  # 64x64 -> 128x128

        x = last(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=x)

        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        # tf.keras.utils.plot_model(self.model, show_shapes=True, to_file='model.png')

    def fit_model(self, epochs):

        images, masks = self.load_images_with_masks()
        aug = Augmentator()

        images = aug.augment_images(images)
        masks = aug.augment_images(masks)

        images = self.normalize_images(images)
        masks = self.normalize_masks(masks)

        images_matrix = self.images_to_4d_array(images, 3)
        masks_matrix = self.images_to_4d_array(masks, 1)

        masks_matrix = np.rint(masks_matrix)

        dataset = tf.data.Dataset.from_tensor_slices((images_matrix, masks_matrix))

        BATCH_SIZE = 1
        SHUFFLE_BUFFER_SIZE = 4 * images_matrix.shape[0]

        print(images_matrix.shape[0])

        dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True).batch(BATCH_SIZE)

        self.init_model()

        model_history = self.model.fit(dataset, epochs=epochs)

        filename = os.path.join(self.save_dir_path, f"save{epochs}.keras")

        if os.path.exists(filename):
            os.remove(filename)

        self.model.save(filename)

        return model_history

    def predict(self, image):
        norm = self.normalize_image(image)
        matrix = self.images_to_4d_array([norm], 3)

        try:
            pred_mask = self.model.predict(matrix, verbose=0)[0]

            pred_mask = np.argmax(pred_mask, axis=-1)

            return pred_mask
        except:
            print("Prediction error!")
            return np.zeros((self.img_size, self.img_size))

    async def predict_async(self, image):
        return self.predict(image)

    async def predict_with_timeout_async(self, image, timeout=1):

        pred_task = asyncio.create_task(self.predict_async(image))

        try:
            async with asyncio.timeout(timeout):
                res = await pred_task
                return res
        except:
            print("Prediction async error!")
            return np.zeros((self.img_size, self.img_size))

    def load_model(self, file_name):
        self.model = keras.models.load_model(os.path.join(self.save_dir_path, file_name))


# asyncio entry point
# for fitting
async def main():
    rec = Recognizer()

    # rec.fit_model(50)

    rec.load_model("save50.keras")
    images, masks = rec.load_images_with_masks()

    for im, mask in zip(images, masks):
        pred = await rec.predict_with_timeout_async(im, 1)

        pred_image = rec.mask_array_to_image_with_background(pred, im)
        show_images_row([im, mask, pred_image])


# start the event loop
asyncio.run(main())
