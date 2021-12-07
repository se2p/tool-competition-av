# How to use model executors

To execute a model please add a custom model executor that needs to implement a `load_model()` and `predict(image)`
function from `model_executor.py`. When running the tool-competition-pipeline use `model` as `executor`!

## Add configs
Add all config settings to `models/config`, if needed this can also be extended and used in a `model_executor`.

## Example
Please check the usage of `models/dave2`!

```
python competition.py
    --visualize-tests
    --time-budget
    60
    --executor
    model
    --beamng-home
    <path_to_beamng>BeamNG.research
    --map-size
    200
    --module-name
    sample_test_generators.one_test_generator
    --class-name
    OneTestGenerator
    --speed-limit
    30
```