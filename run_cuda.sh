#!/bin/bash

docker buildx build . -f ./Dockerfile.cuda --name "ant_colony_cuda" && docker run -it "ant_colony_cuda" --gpus all nvidia/cuda
