# Play with your models

## Prerequisites

The following steps supposed you have already cloned the project into your working directory. If not, follow the [Getting Started](README.md) instructions.

Also, you will need the raw dataset `dataset_multi.csv`. This dataset is not provided in the project (you should have access to). You only need to put this file inside the root `data` directory.


After having followed these instructions, you have normally installed the `smiley` package locally in your conda environment and so, you have at disposal a new command line function: `servier`.

To check if the installation is ready, just ask for the documentation 

```sh
(myenv) $ servier --help
usage: servier [-h] [--kwargs [KWARGS [KWARGS ...]]]
               {train,evaluate,predict} {model1,model2} data_dir save_dir

Arguments for task (training, evaluate, predict)

positional arguments:
  {train,evaluate,predict}
  {model1,model2}       choice of model
  data_dir              location of data directory
  save_dir              location of the directory to save both mlflow runs and
                        model

optional arguments:
  -h, --help            show this help message and exit
  --kwargs [KWARGS [KWARGS ...]]
                        extra args
```

## Training

To train your models, you have two options:

1. Train from the Command Line
2. Train in Python

After having trained your model through one of these two methods, the model will be saved both in MlFlow Tracking (see below) and in the `model` directory which will be used for the [deployement](deploy.md).

For example, if you train the `model1`, you will find in the `model/model/model1` directory a `0` directory containing the model trained.

If you train the two models, the directory will look like this:
```sh
model
├── Dockerfile
├── config.conf
├── models
│   ├── model1
│   │   └── 0
│   │       ├── assets
│   │       ├── saved_model.pb
│   │       └── variables
│   │           ├── variables.data-00000-of-00001
│   │           └── variables.index
│   └── model2
│       └── 0
│           ├── assets
│           ├── saved_model.pb
│           └── variables
│               ├── variables.data-00000-of-00001
│               └── variables.index
```

The `Dockerfile` and `config.conf` file are mandatory to deploy the Tensorflow Server.
### CLI Training

Thanks to the previous package installation, you have access to training through Command Line. Just run:

```sh
(myenv) $ servier train {model} {data_dir} {root_dir}
```

The training will process and the models will be stored both in MlFlow Tracking (see below). Also, the weights of the model will be available inside
### In-Python Training

### Examine your models with MlFlow

The training process is based on MlFlow Tracking Server. If you don't know what MlFlow is and how to use it, dont hesitate to follow [their (strong) documentation](https://mlflow.org/docs/latest/index.html "MlFlow Documentation"). In a few words, MLflow is an open source platform for managing the end-to-end machine learning lifecycle.

Especially, it helps to record and compare parameters and results for differents experiments.

After some training, all the experiments will be stored in the `mlruns` folder. By default, this one is at the root of your project. To explore the runs, juste enter the UI provided with MlFlow (dont hesitate to change the port name because, by default, the port is the same as Flask application)

```sh
(myenv) $ mlflow ui --port 54180
```

The UI looks like the below image, on which you can look for experiments (at the left), and inside experiments, look for the previous runs for this experiment. 
![Mlflow-ui](https://user-images.githubusercontent.com/34337781/107247673-d1c19a80-6a31-11eb-95c1-34caad5f5bf8.png)

For example, for one runs, by clicking on it, you will retrieve the hyperparameters for this run, and the metrics computed during the run:
![Mlflow-run](https://user-images.githubusercontent.com/34337781/107249702-e43cd380-6a33-11eb-8b10-0b31797e3072.png)

A strong utilty provided with MlFlow is the ability to save artifacts and especially model weights. In addition to the model, MlFlow also saved the conda environnement in which the training was made (in the below example, you can see the Tensorflow SavedModel exported in the corresponding artifacts).

![Mlflow-artifacts](https://user-images.githubusercontent.com/34337781/107250377-2cf48c80-6a34-11eb-8d92-4a55808242d9.png)

If you want to deep dive on how MlFlow has been set up and other custom tools to deal with it, dont hesitate to have a look at the reference [Mlflow](mlflow.md).

## Evaluation

### CLI evaluation

### In-Python evaluation

## Prediction

### CLI prediction

### In-python evaluation