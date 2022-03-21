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


class RoadGenerationDiscreteEnv(RoadGenerationEnv):
    """
    Observation:
            Type: MultiDiscrete(2n+1) where n is self.max_number_of_points

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
            Type: MultiDiscrete(4) ?
            Num     Action                  Num
            0       Action type             2
            1       Position                max_number_of_points
            2       New x coord             grid_size * discretization_precision - 2 * safety_buffer
            3       New y coord             grid_size * discretization_precision - 2 * safety_buffer
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

        self.min_oob_percentage = 0
        self.max_oob_percentage = 100

        # state is an empty sequence of points
        self.state = np.empty(self.max_number_of_points, dtype=object)
        for i in range(self.max_number_of_points):
            self.state[i] = (0, 0)  # (0,0) represents absence of information in the i-th cell

        self.viewer = None

        self.discretization_precision = 10
        self.map_buffer_area_width = self.grid_size / 20  # size of the area around the map in which we do not generate
        # points
        number_of_discrete_coords_in_map = self.grid_size * self.discretization_precision
        width_of_buffer_area = self.map_buffer_area_width * self.discretization_precision
        number_of_discrete_coords = number_of_discrete_coords_in_map - 2*width_of_buffer_area

        self.action_space = spaces.MultiDiscrete([2, self.max_number_of_points, number_of_discrete_coords,
                                                  number_of_discrete_coords])

        # create box observation space
        discretized_oob_size = self.max_oob_percentage * self.discretization_precision
        dimensions_list = [number_of_discrete_coords] * (2*max_number_of_points)  # two coords for each point
        dimensions_list.append(discretized_oob_size)
        self.observation_space = spaces.MultiDiscrete(dimensions_list)

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        self.step_counter = self.step_counter + 1  # increment step counter

        action_type = action[0]  # value in [0,1]
        position = action[1]  # value in [0,self.number_of_points-1]
        x = action[2]  # coordinate
        y = action[3]  # coordinate

        logging.info(f"Processing action {str(action)}")

        reward = 0

        if action_type == self.ADD_UPDATE and not self.check_coordinates_already_exist(x, y):
            logging.debug("Setting coordinates for point %d to (%.2f, %.2f)", position, x, y)
            self.state[position] = (x, y)
            reward, max_oob = self.compute_step()
        elif action_type == self.ADD_UPDATE and self.check_coordinates_already_exist(x, y):
            logging.debug("Skipping add of (%.2f, %.2f) in position %d. Coordinates already exist", x, y, position)
            reward = self.invalid_test_reward
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
        obs = self.get_state_observation()
        obs.append(round(max_oob*100*self.discretization_precision))  # max_oob is in [0,1], we make it in 0..1000
        return np.array(obs, dtype=np.float16), reward, done, {}

    def get_state_observation(self):
        obs = [coordinate for tuple in self.state for coordinate in tuple]
        return obs

    def reset(self, seed: Optional[int] = None):
        # super().reset(seed=seed)
        # state is an empty sequence of points
        self.state = np.empty(self.max_number_of_points, dtype=object)
        for i in range(self.max_number_of_points):
            self.state[i] = (0, 0)  # (0,0) represents absence of information in the i-th cell
        # return observation
        obs = self.get_state_observation()
        obs.append(0.0)  # zero oob initially
        return np.array(obs, dtype=np.float16)

    def get_road_points(self):
        road_points = []  # np.array([], dtype=object)
        for i in range(self.max_number_of_points):
            if self.state[i][0] != 0 and self.state[i][1] != 0:
                road_points.append(
                    (
                        (self.state[i][0] + self.map_buffer_area_width) / self.discretization_precision,
                        (self.state[i][1] + self.map_buffer_area_width) / self.discretization_precision
                    )
                )
        logging.debug(f"Current road points: {str(road_points)}")
        return road_points
