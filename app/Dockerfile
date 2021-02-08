FROM continuumio/miniconda3

WORKDIR /code

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
# RUN apk add --no-cache gcc musl-dev linux-headers

COPY . .

RUN conda env create -f conda.yaml

SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

EXPOSE 5000

# Make sure the environment is activated:
RUN echo "Make sure flask is installed:"
RUN python -c "import flask"

# The code to run when container is started:
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "myenv", "flask", "run"]