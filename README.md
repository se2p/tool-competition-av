# GenRL at the Cyber-Physical Systems Testing Competition #

GenRL is a tool that **Gen**erates effective test cases for a lane-keeping system in a simulated
environment using **R**einforcement **L**earning (RL).

This repository is a fork of [tool-competition-av](https://github.com/se2p/tool-competition-av), in which we implemented our RL-based approach.

Install additional dependencies for GenRL with
```
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

pip install -r additional-requirements.txt 
```

To start the test generator using the tool competition pipeline, run `competition.py` with the following command line parameters:
```
--module-name genrl_sbst2022.genrl_test_generator --class-name GenrlTestGenerator 
```

# Cyber-Physical Systems Testing Competition #

The [SBST Workshop](https://sbst22.github.io/) offers a challenge for software testers who want to work with self-driving cars in the context of the usual [tool competition](https://sbst22.github.io/tools/).

## Important Dates

The deadline to submit your tool is: **January 14th 2022**

The results of the evaluation will be communicated to participants on: **February 25th 2022**

The camera-ready paper describing your tool is due to: **Sunday March 18th 2020**

## Goal ##
The competitors should generate virtual roads to test a lane keeping assist system using the provided code_pipeline.

The generated roads are evaluated in a driving simulator. We partnered with BeamNG GmbH which offers a version of their simulators for researchers, named [BeamNG.tech](https://www.beamng.tech/). This simulator is ideal for researchers due to its state-of-the-art soft-body physics simulation, ease of access to sensory data, and a Python API to control the simulation.

[![Video by BeamNg GmbH](https://github.com/BeamNG/BeamNGpy/raw/master/media/steering.gif)](https://github.com/BeamNG/BeamNGpy/raw/master/media/steering.gif)

## Implement Your Test Generator ##
We make available a [code pipeline](code_pipeline) that will integrate your test generator with the simulator by validating, executing and evaluating your test cases. Moreover, we offer some [sample test generators](sample_test_generators/README.md) to show how to use our code pipeline.

## Information About the Competition ##
More information can be found on the SBST tool competition website: [https://sbst22.github.io/tools/](https://sbst22.github.io/tools/)

## Repository Structure ##
[Code pipeline](code_pipeline): code that integrates your test generator with the simulator

[Self driving car testing library](self_driving): library that helps the integration of the test input generators, our code pipeline, and the BeamNG simulator

[Scenario template](levels_template/tig): basic scenario used in this competition

[Documentation](documentation/README.md): contains the installation guide, detailed rules of the competition, and the frequently asked questions

[Sample test generators](sample_test_generators/README.md): sample test generators already integrated with the code pipeline for illustrative purposes 

[Requirements](requirements.txt): contains the list of the required packages.


## License ##
The software we developed is distributed under GNU GPL license. See the [LICENSE.md](LICENSE.md) file.

## Contacts ##

Dr. Alessio Gambi  - Passau University, Germany - alessio.gambi@uni-passau.de

Dr. Vincenzo Riccio  - Universita' della Svizzera Italiana, Lugano, Switzerland - vincenzo.riccio@usi.ch
