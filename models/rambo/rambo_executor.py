from keras.preprocessing.image import img_to_array, load_img
from skimage.exposure import rescale_intensity
from matplotlib.colors import rgb_to_hsv
from PIL import Image
import cv2
import numpy as np
from keras.models import load_model

from models.rambo.config import TestConfig, DataConfig, TrainConfig
from code_pipeline.model_executor import ModelExecutor


class RamboModelExecutor(ModelExecutor):

    def __init__(self, result_folder, time_budget, map_size,
                 oob_tolerance=0.95, max_speed=70,
                 beamng_home=None, beamng_user=None, road_visualizer=None, model_path=None):
        super(RamboModelExecutor, self).__init__(result_folder, time_budget, map_size, oob_tolerance, max_speed,
                                                 beamng_home, beamng_user, road_visualizer, model_path)
        self.prev_image = None
        self.last_images = []
        self.test_config = TestConfig()
        self.data_config = DataConfig()
        self.train_config = TrainConfig()
        self.load_model()
        self.X_train_mean = np.load(self.train_config.X_train_mean_path)

    def process(self, img):
        return self.make_hsv_grayscale_diff_data(img)

    def make_hsv_grayscale_diff_data(self, image, num_channels=2):
        X = np.zeros((2, self.data_config.img_height, self.data_config.img_width, num_channels), dtype=np.uint8)
        prev_image = self.prev_image if self.prev_image is not None else image
        self.prev_image = image
        for j in range(num_channels):
            # img0 = load_img(prev_image, target_size=(self.data_config.img_height, self.data_config.img_width))
            # img1 = load_img(image, target_size=(self.data_config.img_height, self.data_config.img_width))
            img0 = prev_image
            img1 = image
            img0 = img_to_array(img0)
            img1 = img_to_array(img1)
            img0 = rgb_to_hsv(img0)
            img1 = rgb_to_hsv(img1)
            img = img1[:, :, 2] - img0[:, :, 2]
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
            img = np.array(img, dtype=np.uint8)

            X[j, :, :, j] = img
        X = X.astype("float32")
        X -= self.X_train_mean
        X /= 255.0
        return X

    def predict(self, img):
        pil_img = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(pil_img)
        processed = self.process(pil_img)
        prediction = self.model.predict(processed)
        return prediction[0][0]

    def load_model(self):
        self.model = load_model(self.test_config.model_path)
