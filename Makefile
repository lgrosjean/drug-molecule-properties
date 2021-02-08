mlflow:
	mlflow ui --port 54180

lab:
	jupyter lab

build:
	docker-compose build

run:
	docker run --name servier -it servier


mkdocs:
	mkdocs build
	mkdocs serve

start:
	docker-compose up

help:
	cat Makefile