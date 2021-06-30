from code_pipeline.model_executor import ModelExecutor
import tensorflow as tf


class DeepHyperionExecutor(ModelExecutor):

    def __init__(self, result_folder, time_budget, map_size,
                 oob_tolerance=0.95, max_speed=70,
                 beamng_home=None, beamng_user=None, road_visualizer=None, model_path=None, custom_model_path=None):
        super(DeepHyperionExecutor, self).__init__(result_folder, time_budget, map_size, oob_tolerance, max_speed,
                                                   beamng_home, beamng_user, road_visualizer, model_path,
                                                   custom_model_path)
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
        predicted_values = self.model.predict(image, batch_size=1)
        steering_angle = float(predicted_values)
        return steering_angle
