from code_pipeline.model_executor import ModelExecutor
import tensorflow as tf
import numpy as np
import cv2

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 160, 320, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image


def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[80:-1, :, :]  # remove the sky and the car front


def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


class Dave2ModelExecutor(ModelExecutor):

    def __init__(self, result_folder, time_budget, map_size,
                 oob_tolerance=0.95, max_speed=70,
                 beamng_home=None, beamng_user=None, road_visualizer=None, model_path=None):
        super(Dave2ModelExecutor, self).__init__(result_folder, time_budget, map_size, oob_tolerance, max_speed,
                                                 beamng_home, beamng_user, road_visualizer, model_path)
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        self.model: tf.keras.Model = tf.keras.models.load_model(self.model_path)
        # self.model: tf.keras.Model = tf.keras.models.model_from_json(model_path)
        self.model.compile()
        # weights_file = model_path.replace('json', 'hdf5')
        # self.model.load_weights(weights_file)

        # Check model architecture
        self.model.summary()

    def predict(self, image):
        image = np.asarray(image)

        image = preprocess(image)
        image = np.array([image])

        predicted_values = self.model.predict(image, batch_size=1)
        steering_angle = float(predicted_values)
        return steering_angle
