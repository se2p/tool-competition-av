#
# Code taken from https://github.com/udacity/self-driving-car/blob/master/steering-models/evaluation/autumn.py

import scipy.misc
import cv2
import numpy as np
import tensorflow as tf

from keras.models import *
from keras.layers import *

from code_pipeline.model_executor import ModelExecutor


class AutumnModelExecutor(ModelExecutor):

    def __init__(self, result_folder, time_budget, map_size,
                 oob_tolerance=0.95, max_speed=70,
                 beamng_home=None, beamng_user=None, road_visualizer=None, model_path=None):
        super(AutumnModelExecutor, self).__init__(result_folder, time_budget, map_size, oob_tolerance, max_speed,
                                                  beamng_home, beamng_user, road_visualizer, model_path)
        self.model_path = model_path
        self.model = None
        self.cnn = None
        self.fc3 = None
        self.y = None
        self.x = None
        self.keep_prob = None
        self.prev_image = None
        self.last = []
        self.steps = []
        self.load_model()

    def load_model(self):
        # cnn_graph, lstm_json, cnn_weights, lstm_weights
        sess = tf.InteractiveSession()
        saver = tf.train.import_meta_graph("./model-step-2000-val-0.150803.ckpt.meta")
        saver.restore(sess, "./model-step-2000-val-0.150803.ckpt")
        self.cnn = tf.get_default_graph()

        # TODO does this change have any effects?
        # self.fc3 = self.cnn.get_tensor_by_name("fc3/mul:0")
        self.fc3 = self.cnn.get_tensor_by_name("fc3:0")
        self.y = self.cnn.get_tensor_by_name("y:0")
        self.x = self.cnn.get_tensor_by_name("x:0")
        self.keep_prob = self.cnn.get_tensor_by_name("keep_prob:0")

        # with open(self.model_path, 'r') as f:
        #     json_string = f.read()
        # self.model = model_from_json(json_string)
        # self.model.load_weights(

    def process(self, img):
        img = np.asarray(img)
        prev_image = self.prev_image if self.prev_image is not None else img
        self.prev_image = img
        prev = cv2.cvtColor(prev_image, cv2.COLOR_RGB2GRAY)
        next = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        self.last.append(flow)

        if len(self.last) > 4:
            self.last.pop(0)

        weights = [1, 1, 2, 2]
        last = list(self.last)
        for x in range(len(last)):
            last[x] = last[x] * weights[x]

        avg_flow = sum(self.last) / len(self.last)
        mag, ang = cv2.cartToPolar(avg_flow[..., 0], avg_flow[..., 1])

        hsv = np.zeros_like(prev_image)
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return rgb

    def predict(self, img):
        processed_img = self.process(img)
        # cv2.imshow("Flow", img)
        # cv2.waitKey(1)
        image = scipy.misc.imresize(processed_img[-400:], [66, 200]) / 255.0
        cv2.imshow("after scipy", image)
        cv2.waitKey(1)
        cnn_output = self.fc3.eval(feed_dict={self.x: [image], self.keep_prob: 1.0})
        self.steps.append(cnn_output)
        if len(self.steps) > 100:
            self.steps.pop(0)
        output = self.y.eval(feed_dict={self.x: [image], self.keep_prob: 1.0})
        angle = output[0][0]
        return angle
