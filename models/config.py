class ModelsConfig():
    # example inputs for dave2
    model_path = "./self-driving-car-146-2020.h5"
    model_name = "Dave2"
    IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 160, 320, 3
    executor_class_name = "Dave2ModelExecutor"
    module_name = "models.dave2.dave2_executor"
