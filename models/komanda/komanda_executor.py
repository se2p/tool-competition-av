#
# Code taken from https://github.com/udacity/self-driving-car/blob/master/steering-models/evaluation/komanda.py

"""
Udacity self-driving car challenge 2
Team komanda steering model
Author: Ilya Edrenkin, ilya.edrenkin@gmail.com
"""

from collections import deque

import numpy as np

from code_pipeline.model_executor import ModelExecutor
import tensorflow as tf


class KomandaModelExecutor(ModelExecutor):

    def __init__(self, result_folder, time_budget, map_size,
                 oob_tolerance=0.95, max_speed=70,
                 beamng_home=None, beamng_user=None, road_visualizer=None, model_path=None):
        super(KomandaModelExecutor, self).__init__(result_folder, time_budget, map_size, oob_tolerance, max_speed,
                                                   beamng_home, beamng_user, road_visualizer, model_path)
        self.model_path = model_path
        self.model = None
        self.session = None
        self.graph = tf.Graph()
        self.LEFT_CONTEXT = 5  # TODO remove hardcode; store it in the graph
        self.input_images = deque()  # will be of size self.LEFT_CONTEXT + 1
        self.internal_state = []  # will hold controller_{final -> initial}_state_{0,1,2}

        self.load_model()

    def load_model(self):
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(self.model_path)
            # ckpt = tf.train.latest_checkpoint(checkpoint_dir)
        self.session = tf.Session(graph=self.graph)
        # saver.restore(self.session, ckpt)

        # TODO controller state names should be stored in the graph
        self.input_tensors = map(self.graph.get_tensor_by_name,
                                      ["input_images:0", "controller_initial_state_0:0", "controller_initial_state_1:0",
                                       "controller_initial_state_2:0"])
        self.output_tensors = map(self.graph.get_tensor_by_name,
                                       ["output_steering:0", "controller_final_state_0:0", "controller_final_state_1:0",
                                        "controller_final_state_2:0"])

        # Check model architecture
        # self.model.summary()

    def predict(self, img):
        if len(self.input_images) == 0:
            self.input_images += [img] * (self.LEFT_CONTEXT + 1)
        else:
            self.input_images.popleft()
            self.input_images.append(img)
        input_images_tensor = np.stack(self.input_images)
        if not self.internal_state:
            key_list = list(self.input_tensors)
            feed_dict = {key_list[0]: input_images_tensor}
        else:
            feed_dict = dict(zip(self.input_tensors, [input_images_tensor] + self.internal_state))
        steering, c0, c1, c2 = self.session.run(self.output_tensors, feed_dict=feed_dict)
        self.internal_state = [c0, c1, c2]
        return steering[0][0]
