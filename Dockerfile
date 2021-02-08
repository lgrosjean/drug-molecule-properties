FROM continuumio/miniconda3

WORKDIR /usr/src/app

COPY ./ ./

RUN conda env create -f conda.yaml

SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "myenv", "python""]