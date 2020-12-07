# Installation Guide #

## General Information ##
This project contains the code to take part in the tool competition.
It is developed in Python and runs on Windows machines.

## Dependencies ##

### BeamNG simulator ###

This tool needs the BeamNG simulator to be installed on the machine where it is running. 
A free version of the BeamNG simulator for research purposes can be obtained by registering at https://register.beamng.tech and following the instructions provided by BeamNG. 

### Python ###

This code has been tested with Python 3.6

### Other Libraries ###

To easily install the other dependencies with pip, we suggest to create a dedicated virtual environment and run the command:

```pip install -r requirements-36.txt```

Otherwise, you can manually install each required library listed in the requirements-36.txt file.

_Note:_ the version of Shapely should match your system.

### Shapely ###

You can obtain Shapely from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely). 

You should download the wheel file matching you Python version, i.e. download the file with cp36 in the name if you use Python 3.6. The wheel file should also match the architecture of your machine, i.e. you should install the file with either win32 or win_amd64 in the name.

*Example:* if you have a 64 bit machine equipped with Python 3.6, you should download the wheel file named *Shapely‑1.7.1‑cp36‑cp36m‑win_amd64.whl*

To install Shapely, you should run:

```pip install [path of the shapely file]```

## Recommended Requirements ##

[BeamNG](https://wiki.beamng.com/Requirements) recommends the following hardware requirements:
* OS: Windows 10 64-Bit
* CPU: AMD Ryzen 7 1700 3.0Ghz / Intel Core i7-6700 3.4Ghz (or better)
* RAM: 16 GB RAM
* GPU: AMD R9 290 / Nvidia GeForce GTX 970
* DirectX: Version 11
* Storage: 20 GB available space
* Additional Notes: Recommended spec based on 1080p resolution. Installing game mods will increase required storage space. Gamepad recommended.

