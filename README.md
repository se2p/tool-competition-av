# Cyber-Physical Systems Testing Competition #

Starting this year, SBST offers a challenge for software testers who want to work with self-driving cars. 

## Goal ##
The competitors should generate virtual roads to test a lane keeping assist system. 

The generated roads are evaluated in a driving simulator. We partnered with BeamNG GmbH which offers a version of their simulators for researchers, named [BeamNG.research](https://beamng.gmbh/research/). This simulator is ideal for researchers due to its state-of-the-art soft-body physics simulation, ease of access to sensory data, and a Python API to control the simulation.

[![IMAGE ALT TEXT HERE](https://github.com/BeamNG/BeamNGpy/raw/master/media/steering.gif)]

## Implement Your Test Generator ##
We make available a [code pipeline](code_pipeline) that will integrate your test generator with the simulator by validating, executing and evaluating your test cases. Moreover, we offer some [sample test generators](/sample_test_generators) to show how to use our code pipeline.

## Information About the Competition ##
More information can be found on the SBST tool competition website: [https://sbst21.github.io/tools/](https://sbst21.github.io/tools/)

## Repository Structure ##
[Code pipeline](/code_pipeline): code that integrates your test generator with the simulator

[Documentation](/documentation): contains the installation guide and the detailed rules of the competition

[Sample test generators](/sample_test_generators): sample test generators already integrated with the code pipeline for illustrative purposes 

[Requirements](/requirements-36.txt): file containing the list of the required packages

