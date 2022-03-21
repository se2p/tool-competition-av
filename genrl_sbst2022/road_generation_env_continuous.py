import math
import os
import random
import logging
from typing import Optional

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from code_pipeline.executors import MockExecutor
from code_pipeline.tests_generation import RoadTestFactory
from code_pipeline.visualization import RoadTestVisualizer

from road_generation_env import RoadGenerationEnv


class RoadGenerationContinuousEnv(RoadGenerationEnv):
    """
    Observation:
            Type: Box(2n+1) where n is self.number_of_points, the max number of points in the generated roads

            Num     Observation             Min                 Max
            0       x coord for 1st point   self.min_coord      self.max_coord
            1       y coord for 1st point   self.min_coord      self.max_coord
            2       x coord for 2nd point   self.min_coord      self.max_coord
            3       y coord for 2nd point   self.min_coord      self.max_coord
            ...
            2n-2    x coord for 2nd point   self.min_coord      self.max_coord
            2n-1    y coord for 2nd point   self.min_coord      self.max_coord
            n       Max %OOB                0.0                 1.0             # TODO fix

        Actions:
            Type: Box(4) ?
            Num     Action                  Min                 Max
            0       Action type             0                   1
            1       Position                0                   self.number_of_points
            2       New x coord             self.min_coord      self.max_coord
            3       New y coord             self.min_coord      self.max_coord
    """

    ADD_UPDATE = 0
    REMOVE = 1

    def __init__(self, executor, max_steps=1000, grid_size=200, results_folder="results", max_number_of_points=5,
                 max_reward=100, invalid_test_reward=-10):

        super().__init__(executor, max_steps, grid_size, results_folder, max_number_of_points, max_reward,
                         invalid_test_reward)

        self.min_coordinate = 0.0
        self.max_coordinate = 1.0

        self.max_speed = float('inf')
        self.failure_oob_threshold = 0.95

        self.min_oob_percentage = 0.0
        self.max_oob_percentage = 1.0

        # state is an empty sequence of points
        self.state = np.empty(self.max_number_of_points, dtype=object)
        for i in range(self.max_number_of_points):
            self.state[i] = (0, 0)  # (0,0) represents absence of information in the i-th cell

        self.low_coordinates = np.array([self.min_coordinate, self.min_coordinate], dtype=np.float16)
        self.high_coordinates = np.array([self.max_coordinate, self.max_coordinate], dtype=np.float16)
        self.low_observation = np.array([], dtype=np.float16)
        self.high_observation = np.array([], dtype=np.float16)

        self.viewer = None

        # action space as a box
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, self.min_coordinate + 0.1, self.min_coordinate + 0.1]),
            high=np.array([1.0, float(self.max_number_of_points) - np.finfo(float).eps, self.max_coordinate - 0.1,
                           self.max_coordinate - 0.1]),
            dtype=np.float16
        )

        # create box observation space
        for i in range(self.max_number_of_points):
            self.low_observation = np.append(self.low_observation, [0.0, 0.0])
            self.high_observation = np.append(self.high_observation, [self.max_coordinate, self.max_coordinate])
        self.low_observation = np.append(self.low_observation, self.min_oob_percentage)
        self.high_observation = np.append(self.high_observation, self.max_oob_percentage)

        self.observation_space = spaces.Box(self.low_observation, self.high_observation, dtype=np.float16)

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        self.step_counter = self.step_counter + 1  # increment step counter

        action_type = round(action[0])  # value in [0,1]
        position = math.floor(action[1])  # value in [0,self.number_of_points)
        x = action[2]  # coordinate in [self.min_coordinate,self.max_coordinate]
        y = action[3]  # coordinate in [self.min_coordinate,self.max_coordinate]

        logging.info(f"Processing action {str(action)}")

        if action_type == self.ADD_UPDATE and not self.check_coordinates_already_exist(x, y):
            logging.debug("Setting coordinates for point %d to (%.2f, %.2f)", position, x, y)
            self.state[position] = (x, y)
            reward, max_oob = self.compute_step()
        elif action_type == self.ADD_UPDATE and self.check_coordinates_already_exist(x, y):
            logging.debug("Skipping add of (%.2f, %.2f) in position %d. Coordinates already exist", x, y, position)
            reward = -10
            max_oob = 0.0
        elif action_type == self.REMOVE and self.check_some_coordinates_exist_at_position(position):
            logging.debug("Removing coordinates for point %d", position)
            self.state[position] = (0, 0)
            reward, max_oob = self.compute_step()
        elif action_type == self.REMOVE and not self.check_some_coordinates_exist_at_position(position):
            # disincentive deleting points where already there is no point
            logging.debug(f"Skipping delete at position {position}. No point there.")
            reward = self.invalid_test_reward
            max_oob = 0.0

        done = self.step_counter == self.max_steps

        # return observation, reward, done, info
        obs = [coordinate for tuple in self.state for coordinate in tuple]
        obs.append(max_oob)
        return np.array(obs, dtype=np.float16), reward, done, {}

    def reset(self, seed: Optional[int] = None):
        # super().reset(seed=seed)
        # state is an empty sequence of points
        self.state = np.empty(self.max_number_of_points, dtype=object)
        for i in range(self.max_number_of_points):
            self.state[i] = (0, 0)  # (0,0) represents absence of information in the i-th cell
        # return observation
        obs = [coordinate for tuple in self.state for coordinate in tuple]
        obs.append(0.0)  # zero oob initially
        return np.array(obs, dtype=np.float16)


    def get_road_points(self):
        road_points = []  # np.array([], dtype=object)
        for i in range(self.max_number_of_points):
            if self.state[i][0] != 0 and self.state[i][1] != 0:
                road_points.append(
                    (
                        self.state[i][0] * self.grid_size,
                        self.state[i][1] * self.grid_size
                    )
                )
        return road_points
