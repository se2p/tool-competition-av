# Competition Guidelines #

## Goal ##

The tool should generate test inputs to test a Lane Keeping Assist System (LKAS). The competitors should generate roads that force the ego-car, i.e., the test subject, to drive off its lane without creating invalid roads.

## Tests as Driving Tasks ##

In the competition,  tests are driving tasks that the ego-car, i.e., a car equipped with the lane-keeping assist system under test, must complete. These driving tasks are defined in terms of a road that the ego-car must follow. 

### What is a test input for the current competition? ###

For simplicity, we consider driving scenarios on single, flat roads surrounded by green grass. The ego-car must drive along the roads keeping the right lane. The environmental conditions and the road layout are fixed and predefined. In particular, the roads consist of two fixed-width lanes, which are divided by a solid yellow line. Two additional white lines define the exterior boundaries of the lanes.

### What makes a test input? ###

The competitor tools should generate roads as sequences of points, i.e., _road points_, defined in a two-dimensional squared map with predefined size (e.g., 200-by-200 meter). 
The sequence of _road points_ defines the _road spine_, i.e., the road's center line.

The **first point** in the sequence of _road points_ defines the starting location of the ego-car by convention, while the **last point** defines the target location. The road points are automatically interpolated using cubic splines to obtain the final road geometry.

> **NOTE**: the road's geometry automatically defines the initial placement and rotation of the ego-vehicle.

The following image illustrates how a road is defined from the following _road points_ over a map of size 200-by-200 meters: 

```
[
    (10.0, 20.0), (30.0, 20.0), (40.0, 30.0), 
    (50.0, 40.0), (150.0, 100.0), (30.0, 180.0)
]
```

![Sample Road caption="test"](./figures/sample_road.PNG "Sample Road")

In the figure, the inner square identifies the map's boundary (200x200), the white dots correspond to the _road points_, the solid yellow line corresponds to the _road spine_ that interpolates them, and, finally, the gray area is the road.

As the figure illustrates, the road layout consists of one left lane and one right lane (where the car drives). Each lane is four meters wide, and the lane markings are defined similarly to the US standards: solid yellow line in the middle and solid white lines on the side (not drawn in the figure).


### Valid Roads ###

We perform the following validity checks on the roads before using them in the driving tasks:

* Roads must be made of at least 2 _road points_.
* Roads must never intersect or overlap.
* Turns must have a geometry that allows the ego-car to completely fit in the lane while driving on them; so, "too" sharp edges, i.e., turns with small radius, are disallowed.
* Roads must completely fit the given squared map boundaries; implicitly, this limits the roads' maximum length. 
* To avoid overly complex roads and limit the issues with spline interpolation, we also limit the number of _road points_ that can be used to define roads (500/1000 points).

Invalid roads are reported as such (hence not executed), so they do not count as a *failed* test. 

## Competition ##
The contest's organizers provide a [code pipeline](https://github.com/se2p/tool-competition-av/tree/main/code_pipeline) to check the tests' validity, execute them, and keep track of the time budget. The submission should integrate with it.

At the moment, execution can be mocked or simulated. Mocked execution generates random data and is meant **only** to support development. Simulation instead requires executing the BeamNG.research simulation (see the [Installation Guide](INSTALL.md) for details about registering and installing the simulation software).

There's no limit on the number of tests that can be generated and executed. However, there's a limit on the execution time: The generation can continue until the given time budget is reached. The time budget includes time for generating and simulating tests.

To participate, competitors must submit the code of their test generator and instructions about installing it before the official deadline.

## How To Submit ##

Submitting a tool to this competition entails requires you to share your code with us, so the easiest way is to forking the master branch of this repo and send us a pull request with your code in it. Alternatively, you can send the address of a repo where we can download the code or a "tar-ball" with your code in it. 

We will come back to you if we need support to install and run your code.

## Results ##

The test generators' evaluation will be conducted using the same simulation and code-pipeline used for the development. Still, we will not release the test subjects used for the evaluation before the submission deadline to avoid biasing the solution towards it.

For the evaluation we will consider (at least) the following metrics:

- count how many tests have been generated
- count how many tests are valid and invalid
- count how many tests passed, failed, or generated an error. 

> **Note**: tests fail for different reasons. For example, a test fail if the ego-car does not move, or does not reach the end of the road within a timeout (computed over the length of the road), or drives off the lane.

## Sample Test Generators ##
The submission package comes with an implementation of [sample test generators](../sample_test_generators/README.md). This serves the dual purpose of providing an example on how to use our code pipeline, and a baseline for the evaluation.

## Installation ##
Check the [Installation Guide](INSTALL.md)

## Technical considerations ##
The competition code can be run by executing `competition.py` from the main folder of this repo.

Usage(from command line): 

```
competition.py [OPTIONS]

Options:
  --executor [mock|beamng]
  --time-budget INTEGER     [required - time budget in seconds]
  --map-size INTEGER        [default 200, as 200x200]
  --module-name TEXT        [required - name of your test generation module]
  --module-path TEXT        [path to your test generation module]
  --class-name TEXT         [required - name of the class implementing your generator]
  --visualize-tests         [visualize the last generated test].
  --beamng-home TEXT        [considered only if beamng executor is selected]
  --help                    Show this message and exit.
```

For example, to execute the `one-test-generator.py`, activate your virtual environment (if you have one... and you should), then `cd` to the root of this repository and run:

``` 
python competition.py \
        --visualize-tests \
        --time-budget 10 \
        --executor mock \
        --map-size 200 \
        --module-name sample_test_generators.one_test_generator \
        --class-name OneTestGenerator
```
