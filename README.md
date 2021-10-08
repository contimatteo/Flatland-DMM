# DMM Flatland

You can find the full report [here](./docs/report/main.pdf).

You can find the summary slides [here](./docs/pitch/main.pdf).

## Challenge

![](https://i.imgur.com/9cNtWjs.gif)

https://www.aicrowd.com/challenges/flatland-challenge

## Installation

This project was tested only with the following `python` versions:

```
$ Python 3.7.10
$ Python 3.8.5
```

for installing the dependencies:

```
$ pip install -r requirements_cpu.txt
```

### OS Errors

in case you have some errors during the rendering function, please run:

```
$ pip install -U pyglet==1.5.11
```

for more information about this problem look [here](https://github.com/openai/gym/issues/2101).

## Configuration File

The base structure of the json configurations file can be found inside [this file](./src/configs/run.json).

Look at the [examples](./examples) folder for more advanced configurations.

## Usage

```
$ python src/run.py -h
    usage: Flatland-DMM [options]

    Executable for training/testing the behaviour of an arbitrary number of trains inside the Flatland rail environment.

    optional arguments:
    -h, --help       show this help message and exit
    --config CONFIG  path of the json file with the running configurations.
```

for running the program with the default configurations:

```
$ python src/run.py
```

for running the program with one the pre-built examples:

```
$ python src/run.py --config "./examples/agents/ddqn.json"
```

if you want to run the program with your custom configurations:

```
$ python src/run.py --config "<your_file_path>.json"
```

## Authors

- **Matteo Conti** - [GitHub](https://github.com/contimatteo)
- **Davide Sangiorgi** - [GitHub](https://github.com/DavideSangiorgi)
- **Manuel Mariani** - [GitHub](https://github.com/manuel-mariani)

See also the list of [contributors](https://github.com/contimatteo/Flatland-DMM/graphs/contributors) who participated in this project.
