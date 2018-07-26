# SC2CombatPredictor

The SC2CombatPredictor is a toolchain that supports

* simulating army battles in StarCraft II using a custom map 
* automating the simulation using a shell script provided
* predicting battle outcomes via machine learning

## Introduction

This is merely a branch of the SC2CombatPredictor implemented by lschmelzeisen. 
In order to execute the given files you need to follow his [instructions](https://github.com/lschmelzeisen/SC2CombatPredictor#installation).

## Usage 

The whole process is divided into three steps to generate, process and learn from the data.

### Combat Generator

The combat_generator executes an infinite loop that calls the provided custom map: 
CombatGenerator-v1_1
This map simulates 25 battles with random armies. Those battles are saved in ./replays/*.SC2Replay

To execute the combat_generator cd into the SC2CombatPredictor-repository and execute
```sh
$ python -m bin.combat_generator
```

The combat_generator may throw a socket error. As a workaround there is a shell-file called exec.sh. 
The purpose of this script is to call the python-file [`combat_generator`](https://github.com/FrankSchaust/SC2CombatPredictor/blob/master/bin/combat_generator.py).
The downside is that is can't cancel the blizzard error reports, so that you have to cancel them manually.

You can call it by simply cd-ing into the SC2CombatPredictor-repository and execute
```sh
$ ./exec.sh
```

### Combat Observer

The observer transforms the replay files *.SC2Replay into *.SC2Replay_parsed and saves them at ./replays_parsed. Those files contain the starting constellations and their respective labels. 
The replays mays throw an error because of wrong parsing. At the moment those files are deleted automatically and the observation continues with the next file.

To use the observer, make sure there is at least one replay in the ./replays folder and execute:
```sh
$ python -m bin.combat_observer
```

### Combat Learner v2

The combat learner reads the data from all *.SC2Replay_parsed files within the ./replays_parsed folder. 
Its task is to convert them into array to use for the learning algorithms and then execute the algorithm itself. 

To execute the learning algorithm, make sure there is at least one *.SC2Replay_parsed file in the ./replays_parsed folder and execute:
```sh
$ python -m bin.combat_learner_v2
```

### Replay Viewer

The replay viewer lets you watch the generated *.SC2Replay files without any computation in the background. So if you want to get a sense of how these replays look in-game you can execute:
```sh
$ python -m bin.replay_view
``` 

