#!/bin/bash

docker buildx build . -f ./Dockerfile.rocm -t "ant_colony_rocm" && docker run -it --device=/dev/kfd --device=/dev/dri --group-add video "ant_colony_rocm"
