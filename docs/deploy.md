# Deploy your application

## Prerequisites

You will need two software to install with Docker tools, if you havent them already:

- [`docker`](https://docs.docker.com/engine/installation/ "Docker homepage") (obviously)
- [`docker-compose`](https://docs.docker.com/compose/install/ "Docker Compose homepage")

Docker is an interesing tools, but a bit complex to explain in a few words. If you follow the following steps, everything will be ready.

You will also need your two models trained. If you havent trained them already, you can dowload the pre-trained model directly and unzip them in the `model` folder. The zip file can be downloaded from the GitHub release page: https://github.com/lgrosjean/drug-molecule-properties/releases/tag/v0.0.2 (or directly from [this link](https://github.com/lgrosjean/drug-molecule-properties/files/5945779/models.zip "Models")).

You can also train them directly through command-line with the `smiley` package previously installed.

## Prepare the image

At the root of your project directory, run  `make build` (or `docker-compose build`) to build your docker images. That may take some time but is only required once. 

At the end of the process will create two images, that will be the base for your experiments:

- `flask-app`: container containing the requirements to deploy the Flask application
- `tf-models`: container containing the requirements to deploy the two Tensorflow trained models (Follow the [documentation here](https://www.tensorflow.org/tfx/serving/docker "Tensorflow Serving Tutorial") if you havent try yet)

You can confirm that looking on results of `docker images` command.

```shell
$ docker images
REPOSITORY               TAG          IMAGE ID       CREATED        SIZE
flask-app                latest       0bf5daf6e14f   4 hours ago    2.61GB
tf-models                latest       925b5e04cd7e   4 hours ago    300MB
```

The `flask-app` container is a bit large (2.6Go) because it contains all the requirements for the conda environnement behind, to use the `rdkit` package.

## Run the images

Run `make run` (or just `docker-compose up`) to start the Flask application inside the container (named `flask_server`) and the Tensorflow Serving server (named `tf-serving`). 

Then, just go to the pointed URL in your command line (or just http://localhost:5000) and you're ready to play with the Flask application and the two trained models to predict molecule properties!

![Docker](https://user-images.githubusercontent.com/34337781/107245943-02a0d000-6a30-11eb-875e-7d02d9a0bc71.png)

## Play with your application

After having deploy both of your application (Flask and Tensorflow Serving), two ports have been set up to let you play with all the work.

1. Port 5000:  Flask Application
2. Port 8501: Tensorflow Serving

### Flask Application

The layout of the application is pretty simple and consist in two page:

1. An input page
2. A prediction page

The input page let you input a Smile in a string format, chose one of the two trained model, and then predict. 


![flask-input](https://user-images.githubusercontent.com/34337781/107254602-bf4a5f80-6a37-11eb-898a-fc8c4ad7522a.png)

The prediction page will return the raw prediction from the TensorFlow Serving server.

![flask-output](https://user-images.githubusercontent.com/34337781/107254711-dbe69780-6a37-11eb-8929-95c138361b96.png)

### Tensorflow Serving Server

As soon as the Docker image for your Tensorflow models has been deployed, you have access to prediction through two dedicated API automatically deployed by Tensorflow Serving, one for each trained model.

| Model  | API endpoint                                   |
| ------ | ---------------------------------------------- |
| model1 | http://localhost:8501/v1/models/model1:predict |
| model2 | http://localhost:8501/v1/models/model2:predict |

Tensorflow Serving is also in charge to create the endpoint `predict` on which you can make also predictions, without using the Flask application.

=== "model1"

    ```python
    >>> import requests
    >>> from smiley.learner import Model1Learner
    >>> learner_1 = Model1Learner()
    >>> inputs = learner_1.X.iloc[0].values.tolist()
    >>> url = "http://localhost:8501/v1/models/model1:predict"
    >>> payload = {'instances': inputs}
    >>> res = requests.post(url, json=payload)
    >>> print(res.json())
    {'predictions': [[0.990148425]]}
    ```

=== "model2"

    ```python
    >>> import requests
    >>> from smiley.learner import Model2Learner
    >>> learner_2 = Model1Learner()
    >>> inputs_2 = learner_2.X.iloc[0]
    >>> inputs_2 = [inputs]
    >>> url_2 = "http://localhost:8501/v1/models/model2:predict"
    >>> payload_2 = {'instances': inputs_2}
    >>> res_2 = requests.post(url_2, json=payload_2)
    >>> print(res_2.json())
    {'predictions': [[0.501264036]]}
    ```


