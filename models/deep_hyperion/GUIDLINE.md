# How to use this model

To execute a model please add a custom model executor that needs to implement a `load_model()` and `predict(image)`
function from `model_executor.py`.

Example usage with one of the models in `models/deep_hyperion`, please add the full path to the project folder:
```
--visualize-tests
--time-budget
60
--executor
model
--beamng-home
C:\Users\tobia\Documents\BeamNG.research
--map-size
200
--module-name
sample_test_generators.one_test_generator
--class-name
OneTestGenerator
--speed-limit
30
--model_path
<<path_to_project_folder>>\models\deep_hyperion\self-driving-car-178-2020.h5
```