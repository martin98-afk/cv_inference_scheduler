#!/bin/bash
RUNNING_PATH=$(cd "$(dirname $0)"; pwd)

function load_settings() {
    CONTAINER_NAME="cv-models-inference-scheduler"
    CONTAINER_IMAGE="cv-inference:v1.1"
    CONTAINER_RESTART="always"
    CONTAINER_NETWORK="host"
    CONTAINER_CPUS="8"
    CONTAINER_MEMORY="16g"
    CONTAINER_SHM_SIZE="3g"
}
load_settings

function install_service() {
    docker run -d --name=${CONTAINER_NAME} \
        --restart=${CONTAINER_RESTART} \
        --network=${CONTAINER_NETWORK} \
        --cpus=${CONTAINER_CPUS} \
        --memory=${CONTAINER_MEMORY} \
        --shm-size=${CONTAINER_SHM_SIZE} \
        --gpus all \
        --runtime=nvidia \
        --privileged=true \
        -v /etc/localtime:/etc/localtime:ro \
        -v ${RUNNING_PATH}:/app \
        ${CONTAINER_IMAGE} \
        python -u start_object_detection_server.py|| exit 1
}
install_service
