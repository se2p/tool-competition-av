import logging

import gym
from stable_baselines3 import PPO, A2C

from stable_baselines3.common.env_checker import check_env

from code_pipeline.beamng_executor import BeamngExecutor
from code_pipeline.executors import MockExecutor
from code_pipeline.visualization import RoadTestVisualizer
from road_generation_env import RoadGenerationEnv
from road_generation_env_continuous import RoadGenerationContinuousEnv
from road_generation_env_discrete import RoadGenerationDiscreteEnv
from road_generation_env_transform import RoadGenerationTransformationEnv

logging.basicConfig(level=logging.DEBUG)

# test_executor = MockExecutor(result_folder="results", time_budget=1e10, map_size=200,
#                                      road_visualizer=RoadTestVisualizer(map_size=200))

test_executor = BeamngExecutor(generation_budget=10000, execution_budget=10000, time_budget=10000,
                               result_folder="results", map_size=200, beamng_home="D:\\BeamNG",
                               beamng_user="D:\\BeamNG_user\\", road_visualizer=RoadTestVisualizer(map_size=200))

# env = RoadGenerationContinuousEnv(test_executor, max_number_of_points=20)
# env = RoadGenerationDiscreteEnv(test_executor, max_number_of_points=8)
env = RoadGenerationTransformationEnv(test_executor, max_number_of_points=4)

# Instantiate the agent
model = PPO('MlpPolicy', env, verbose=1, batch_size=2)
model.learn(total_timesteps=int(2),  log_interval=1)

# check_env(env)


# Enjoy trained agent
# obs = env.reset()
# for i in range(100):
#     action = env.action_space.sample()
#     print(str(action))
#     obs, rewards, dones, info = env.step(action)
#     print(f"Lunghezza strada: {len(env.get_road_points())}")
#     logging.debug(f"Observation is {str(obs)}")
#     #env.render()
