#!/bin/bash

docker build --file Dockerfile --tag wavenet .
docker run -it --gpus all --ipc=host -p 8080:8080 -v /home/$USER/wavenet-vocoder/:/home/$USER wavenet:latest bash
