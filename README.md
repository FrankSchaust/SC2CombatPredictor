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

### Data structure and Files

The data system is divided into four separate folders. Every folder contains subfolders that specify the map-version this data comes from. 

/replays is the folder, that contains unparsed replays. 
/parsed_basic contains all replay-files that were already parsed. Every parsed replays-file from /replays will automatically move to /parsed_basic 

/log contains .csv-files which contain a list of both armies and their unit ids as well as an integer which determines the winning side
the files in this folder are used to get the supply difference for each match and to have an easy accessible and fast to read source for the baseline, which only uses unit ids for evaluation
/Proc_Data_ contains a Layer.csv file for every match. The files are separated by unique file paths. 
The Layer.csv files are the files that are read by the learner. They contain 91731 lines with one number each, where the last the numbers represent the one-hot vector as ground-truth for the learning algorithm.

### Combat Generator

The combat_generator executes an infinite loop that calls the provided custom map: 
CombatGenerator-v1_3d_15sup
This map simulates 25 battles with random armies. Those battles are saved in ./replays/*.SC2Replay

You can change the used map by changing the constant REPLAY_VERSION in lib/config.py
The script searches for the map by searching for CombatGenerator-v + the specified version. 
If you want to create your own maps make sure to stick to the naming scheme, or change the path in the combat_generator.py code directly.

The combat_generator may throw a socket error. This error is caught by a try and catch command, but be aware that roughly 50% of the runs will be useless. If you run other programs on your workstation while simulating the battles, the error rate might be higher. Also the catch block can't cancel the blizzard error reports you receive when the game crashes, so that you have to cancel them manually.

To execute the combat_generator cd into the SC2CombatPredictor-repository and execute
```sh
$ python -m bin.combat_generator
```

### Combat Observer

The observer transforms the replay files *.SC2Replay into *.csv and saves them at ./Proc_Data_. Those files contain the starting constellations and their respective labels. 

To use the observer, make sure there is at least one replay in the ./replays folder and execute:
```sh
$ python -m bin.combat_observer_2
```

### Combat Learner

The combat learner reads the data from all specified versions from the Proc_Data_ folder.
At first it converts the .csv files into a TextLineDataset from which it feeds the learning algorithm. 

You can configure the training as follows: 
Line 28 defines an array with all versions to use for this training
Lines 30 to 32 allow to define the batch size to use, the number of samples that are going to evaluated and the number of epochs which the training should progress.
Line 34 data_augmentation is a bool which lets you decide whether you want to augment your data or not.
Line 36 defines the trasin/test-split
In line 42 you define the learning rate
Line 45 lets you define the directory in which the tensorboard logs and the model are saved. If you want to continue training an existing model, this directory has to point to the existing models directory!

Line 116 to 119 contains the model declaration. The models are defined in separate .py files so that you can easily swap the model functions to test other models as well. 

Line 151 lets you define the last stop for your model. This is None if you want to train a model from scratch, but if you stopped training at a certain epoch and you want to resume your training, just set last_epoch to the epoch you want to continue.

At every finished epoch the learner will calculate loss, accuracy, precision, recall and f1-score as well as the time spent for this epoch and log it as an event file in the tensorboard-directory defined in line 45

To execute the learning algorithm, make sure there is at least one *.SC2Replay_parsed file in the ./replays_parsed folder and execute:
```sh
$ python -m bin.learner
```

### Analyzer

The analyzer is able to analyze a specified model with respect to the supply difference of the battles.
You can specify the model in Line 30 with the variable tensorboard_dir. Set the directory to the directory of the model and specify the epoch you want to analyze in line 97.
To specify the samples that are used to evaluate your models, you can set the variables cap and scap to the start and end of the sample range. 

The analyzer will log the accuracy with respect to the supply difference of the samples and will print a classification report with respect to the 3 labels into the console.

```sh
$ python -m bin.analyzer
``` 

