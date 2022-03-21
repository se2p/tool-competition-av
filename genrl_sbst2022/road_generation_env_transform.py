import math
import os
import random
import logging
from typing import Optional
from collections import deque

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from code_pipeline.executors import MockExecutor
from code_pipeline.tests_generation import RoadTestFactory
from code_pipeline.visualization import RoadTestVisualizer

from genrl_sbst2022.road_generation_env import RoadGenerationEnv


class RoadGenerationTransformationEnv(RoadGenerationEnv):
    """
    Start with a random sequence of points, and with each action we modify it slightly
    Observation:
            Type: Box(2n+1) where n is self.max_number_of_points

            Num     Observation             Min                 Max
            0       x coord for 1st point   self.min_coord      self.max_coord
            1       y coord for 1st point   self.min_coord      self.max_coord
            2       x coord for 2nd point   self.min_coord      self.max_coord
            3       y coord for 2nd point   self.min_coord      self.max_coord
            ...
            2n-2    x coord for 2nd point   self.min_coord      self.max_coord
            2n-1    y coord for 2nd point   self.min_coord      self.max_coord
            n       Max %OOB                0.0                 1.0

        Actions:
            Type: MultiDiscrete(4) ?
            Num     Action                  Num
            0       Action type             4                       # move up, down, left, right
            1       Position                max_number_of_points
            2       Amount                  3                       # small, medium, high amount of movement?
    """

    MOVE_UP = 0
    MOVE_RIGHT = 1
    MOVE_DOWN = 2
    MOVE_LEFT = 3

    def __init__(self, executor, max_steps=1000, grid_size=200, results_folder="results", max_number_of_points=5,
                 max_reward=100, invalid_test_reward=-10):

        super().__init__(executor, max_steps, grid_size, results_folder, max_number_of_points, max_reward,
                         invalid_test_reward)

        self.min_coordinate = 0.0
        self.mid_coordinate = 0.5
        self.max_coordinate = 1.0
        self.safety_buffer = 0.1

        self.max_speed = float('inf')
        self.failure_oob_threshold = 0.95

        self.min_oob_percentage = 0.0
        self.max_oob_percentage = 100.0

        self.low_observation = np.array([], dtype=np.float16)
        self.high_observation = np.array([], dtype=np.float16)

        # state is an empty sequence of points
        self.state = np.empty(self.max_number_of_points, dtype=object)
        for i in range(self.max_number_of_points):
            self.state[i] = (
                random.uniform(self.min_coordinate + self.safety_buffer, self.max_coordinate - self.safety_buffer),
                random.uniform(self.min_coordinate + self.safety_buffer, self.max_coordinate - self.safety_buffer)
            )

        self.viewer = None

        # create box observation space
        for i in range(self.max_number_of_points):
            self.low_observation = np.append(self.low_observation, [0.0, 0.0])
            self.high_observation = np.append(self.high_observation, [self.max_coordinate, self.max_coordinate])
        self.low_observation = np.append(self.low_observation, self.min_oob_percentage)
        self.high_observation = np.append(self.high_observation, self.max_oob_percentage)

        self.observation_space = spaces.Box(self.low_observation, self.high_observation, dtype=np.float16)

        # create action space
        self.action_space = spaces.MultiDiscrete([4, self.max_number_of_points, 3])
        self.change_amounts = [0.025, 0.05, 0.25]  # corresponding to 5, 10 points on the map
        self.change_amounts_names = ["low", "medium", "high"]
        self.action_names = ["MOVE UP", "MOVE_RIGHT", "MOVE_DOWN", "MOVE_LEFT"]

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        self.step_counter = self.step_counter + 1  # increment step counter

        action_type = action[0]  # value in [0,1]
        position = action[1]  # value in [0,self.number_of_points-1]
        amount = action[2]  # value in [0,2] for small, medium, high amounts of movement

        logging.info(f"Processing action {str(action)}")

        self.state[position], is_valid = self.process_action(action_type, position, amount)

        done = False

        if is_valid:
            logging.info("Action was valid, computing step.")
            reward, max_oob = self.compute_step()
        else:
            reward = self.invalid_test_reward
            max_oob = 0.0
            done = True  # episode ends if an invalid road is produces

        # episode ends after max number of steps per episode is reached or a failing test is produced
        if self.step_counter == self.max_steps or reward == self.max_reward:
            done = True

        # return observation, reward, done, info
        obs = self.get_state_observation()
        obs.append(max_oob)  # append oob to state observation to get the complete observation
        return np.array(obs, dtype=np.float16), reward, done, {}

    def get_state_observation(self):
        obs = [coordinate for tuple in self.state for coordinate in tuple]
        return obs

    def reset(self, seed: Optional[int] = None):
        # super().reset(seed=seed)
        self.reset_state()
        # return observation
        obs = self.get_state_observation()
        obs.append(0.0)  # zero oob initially
        return np.array(obs, dtype=np.float16)

    def get_road_points(self):
        road_points = []
        for i in range(self.max_number_of_points):
            if self.state[i][0] != 0 and self.state[i][1] != 0:
                road_points.append(
                    (
                        self.state[i][0] * self.grid_size,
                        self.state[i][1] * self.grid_size
                    )
                )
        return road_points

    def process_action(self, action_type, position, amount):
        old_point = self.state[position]
        change_amount = self.change_amounts[amount]
        x_change_amount = 0
        y_change_amount = 0

        logging.debug(f"Processing action {self.action_names[action_type]} ({self.change_amounts_names[amount]}) on "
                      f"position {position}, with current value ({old_point[0]}, {old_point[1]})")

        if action_type == self.MOVE_UP:
            y_change_amount = change_amount
        elif action_type == self.MOVE_DOWN:
            y_change_amount = -1 * change_amount
        elif action_type == self.MOVE_RIGHT:
            x_change_amount = change_amount
        elif action_type == self.MOVE_LEFT:
            x_change_amount = -1 * change_amount

        new_point = (
            old_point[0] + x_change_amount,
            old_point[1] + y_change_amount
        )
        min_admissible_coord = self.min_coordinate + self.safety_buffer
        max_admissible_coord = self.max_coordinate - self.safety_buffer
        x = new_point[0]
        y = new_point[1]
        if min_admissible_coord <= x <= max_admissible_coord and min_admissible_coord <= y <= max_admissible_coord:
            logging.debug(f"Position {position} changed from ({old_point[0]}, {old_point[1]}) to ({x}, {y})")
            return new_point, True
        else:
            logging.debug(f"Invalid action, tried changing from ({old_point[0]}, {old_point[1]}) to ({x}, {y})")
            return old_point, False

    def reset_state(self):
        logging.info("Resetting state")
        if self.max_number_of_points != 4:
            self.state = np.empty(self.max_number_of_points, dtype=object)
            for i in range(self.max_number_of_points):
                self.state[i] = (
                    random.uniform(self.min_coordinate + self.safety_buffer, self.max_coordinate - self.safety_buffer),
                    random.uniform(self.min_coordinate + self.safety_buffer, self.max_coordinate - self.safety_buffer)
                )
        else:
            # if we have exactly four points, generate one of them in each quadrant (to reduce initially invalid roads)
            # TODO: we should generalize this (both to work with any number of points)
            point_q1 = (
                random.uniform(self.mid_coordinate + self.safety_buffer, self.max_coordinate - self.safety_buffer),
                random.uniform(self.mid_coordinate + self.safety_buffer, self.max_coordinate - self.safety_buffer)
            )
            point_q2 = (
                random.uniform(self.min_coordinate + self.safety_buffer, self.mid_coordinate - self.safety_buffer),
                random.uniform(self.mid_coordinate + self.safety_buffer, self.max_coordinate - self.safety_buffer)
            )
            point_q3 = (
                random.uniform(self.min_coordinate + self.safety_buffer, self.mid_coordinate - self.safety_buffer),
                random.uniform(self.min_coordinate + self.safety_buffer, self.mid_coordinate - self.safety_buffer)
            )
            point_q4 = (
                random.uniform(self.mid_coordinate + self.safety_buffer, self.max_coordinate - self.safety_buffer),
                random.uniform(self.min_coordinate + self.safety_buffer, self.mid_coordinate - self.safety_buffer)
            )
            d = deque([point_q1, point_q2, point_q3, point_q4])
            if random.choice([True, False]):
                d.reverse()  # make the points go "clockwise"
            d.rotate(random.randint(0, 3))  # optionally shift the starting point
            self.state = np.array(d, dtype=object)  # convert deque to np array
