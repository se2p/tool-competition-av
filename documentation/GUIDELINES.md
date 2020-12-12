# Competition Guidelines #

## Goal ##

The tool should generate test inputs to test a Lane Keeping Assist System (LKAS). The competitors should force the test subject to drive of lane, without creating invalid roads.

## Test Inputs ##

A test input for the considered subject is a scenario. A scenario consists of a car equipped with a test lane keeping assist system, a road, a driving task and the environment definition. The driving task is to drive from road start to road end without going astray.

### What is a test input for the current competition? ###

For the sake of simplicity, we consider scenarios consisting of single plain asphalt roads surrounded by green grass on which the car has to drive keeping the right lane. The environment is always set to a clear day without fog. The roads are composed of two lanes with fixed width in which there is a yellow center line plus two white lines that separate each lane from the non-drivable area.

### How Should I Generate Test Inputs? ###

The competitor tool should generate roads as sequences of points in a two-dimensional space inside the boundaries of a predefined map.

In fact, roads are defined by a ordered sequence of coordinates in a two-dimensional space, i.e., (x, y). Each point corresponds to the central line of the road, a.k.a., *road spine*. 

The first point in the sequence defines the starting location the last one the target location.

> **NOTE**: the initial placement and rotation of the ego-vehicle is automatically defined.

Test generators must generate sequences of coordinates, i.e., *road points*, that define the overall geometry of the roads. The road points are automatically interpolated using cubic splines to obtain the final road geometry.

The following image illustrates how a road is rendered from the following *road points*: 
`[(10,20), (30, 20), (40, 30), (50, 40), (150, 100), (30, 180)]`

![Sample Road caption="test"](./figures/road_sample.pdf "Sample Road")

In the figure, the inner square identifies the boundary of the map (200x200), the white dots are the *road points*, the yellow solid line is the *road spine* that interpolates them, and, finally, the gray area is the road.

As the figure illustrates, the road layout consist of one left lane and one right lane (where the car drives). Each lane is 4 meter wide and the lane markings are defined according to the US standard (solid yellow line in the middle and solid white lines on the side).

> **Note**: road material, slope, layout, and environmental factors such as weather and time-of-day are fixed and predefined  (daylight, no rain, clear-sky, green grass, no curbs, cement/asphalt/concrete). 

### Valid Roads ###

Roads must be valid. An invalid input results in an invalid test.

> **NOTE**: An *invalid* test does not count as a *failed* test.

We perform the following validity checks before executing a test:
* We do no allow intersecting or overlapping roads.
* Turns must have a geometry that allows car to drive on them, so "too" sharp edges are disallowed (limit on curvature). 
* To limit the length of road, roads must be completely within a predefined squared map, so we enforce a map limit (box of given size, the road should not cross the limits), and have a maximum length. 
* Roads must be made of at least 4 points.
* To avoid overly complex roads we also limit the points that can be used to define them (500/1000 points).


## Competition ##
The contest's organizers provide a [code pipeline](../../code_pipeline/) to check the validity of the tests, to execute them, and to keep track of the time budget. The submission should integrate with it.

Execution can be mocked or simulated. 

Simulation requires registering and installing the simulation software from BeamNG GmbH (see the [Installation Guide](INSTALL.md)). Otherwise, if the mock is used, an approximation of the simulation time will be computed. 

There's no limit on the number of test that can be generated and executed. The time budget will be up to one hour. The generation will continue until a given time budget is reached. The time budget includes time for generating and simulating tests.

Competitors must submit (1) the code of the test generator, (2) instructions about the installation of the test generator, (3) a test suite that consists of roads in the agreed format. 

## How To Submit ##

Submitting a tool to this competition will mean:
* Creating a GitHub account with your public name
* Forking the master branch of our repo into your own local branch
* Integrating your test generator into your repo
* Commit your branch
* Submit a pull request back to master

## Results ##
To evaluate submissions we will follow those guidelines:
- count how many valid/invalid tests have been executed
- count how many tests passed/failed. A test fail if the test subject does not reach the end of the road within a timeout (computed over the length of the road) or drives off the lane/road

The evaluation will be conducted using the same simulation, BeamNG.tech. The test subjects used for the evaluation will be not released before the submission deadline to avoid biasing the solution towards it. Mock test subjects are provided with the package.


## Sample Test Generators ##
The submission package comes with an implementation of [sample test generators](./sample_test_generators). This serves the dual purpose of providing an example on how to use our code pipeline, and a baseline for the evolution.

## Installation ##
Check the [Installation Guide](INSTALL.md)

## Technical considerations ##
The competiton code can be run by launching competition.py in the main folder.

Usage(from command line): 

```
competition.py [OPTIONS]

Options:
  --executor [mock|beamng]
  --beamng-home TEXT
  --time-budget INTEGER     [required]
  --map-size INTEGER
  --module-name TEXT        [required]
  --module-path TEXT
  --class-name TEXT         [required]
  --visualize-tests         Visualize the last generated test.
  --help                    Show this message and exit.
 ```
