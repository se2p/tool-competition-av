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
