import logging as log

from stable_baselines3 import PPO

from genrl_sbst2022.road_generation_env_transform import RoadGenerationTransformationEnv

class GenrlTestGenerator:
    """
        Generates tests using a RL-based approach
    """

    def __init__(self, executor=None, map_size=None):
        self.executor = executor
        self.map_size = map_size

    def start(self):
        log.info("Starting CaRL test generator")

        # Instantiate the environment
        # env = RoadGenerationContinuousEnv(test_executor, max_number_of_points=20)
        # env = RoadGenerationDiscreteEnv(test_executor, max_number_of_points=8)
        env = RoadGenerationTransformationEnv(self.executor, max_number_of_points=4)

        # Instantiate the agent
        model = PPO('MlpPolicy', env, verbose=1)

        # Start training the agent
        log.info("Starting training")
        model.learn(total_timesteps=int(1e2))

        # If training is done and we still have time left, we generate new tests using the trained policy until the
        # given time budget is up.
        log.info("Generating tests with the trained agent")
        while not self.executor.time_budget.is_over():
            obs = env.reset()
            while not done:
                action = model.predict(observation=obs)
                obs, reward, done, info = env.step(action)
