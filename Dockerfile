FROM continuumio/miniconda3

WORKDIR /usr/src/app

COPY ./ ./

RUN conda env create -f conda.yaml

RUN /bin/bash -c "source activate mmyenv"