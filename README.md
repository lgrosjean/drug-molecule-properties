# :pill: Drug Molecule Properties Forecasting

Project to predict drug molecules properties with Deep Learning

## Quick start

### Installation

*This process supposes that you have [Anaconda](https://www.anaconda.com/distribution/ "Anaconda homepage") (or [Miniconda](https://docs.conda.io/en/latest/miniconda.html "Miniconda homepage")) already installed, with [git](https://git-scm.com/downloads "Git download page").*

To install current project locally and access it, first clone the project (do not copy the `$` symbol, it only indicates that they are terminal commands):  
```shell
$ git clone https://github.com/lgrosjean/drug-molecule-properties.git
$ cd drug-molecule-properties
```

Next, run the following to create Anaconda environment:
```shell
$ conda create -y --name myenv python=3.6
$ conda activate myenv
```

Then, install depencies:
```shell
(myenv) $ pip install -r requirements.txt
```

Finally, build local project into a propre Python package:
```shell
$ pip install . -U
```

*Check your installation*

```shell 
$ python
```
```python
>>> import drug_prophet
>>> print(drug_prophet.__version__)
"0.0.1"
```


### Use with Docker

https://github.com/ageron/handson-ml2/tree/master/docker