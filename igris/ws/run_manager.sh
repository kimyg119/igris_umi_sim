#!/bin/bash
set -e

CONFIG_ROOT=${IGRIS_C_CONFIG_ROOT:-/robros/configs}
ROBOT_CONFIG_FILE=${IGRIS_C_CONFIG_FILE:-${CONFIG_ROOT}/robot.yaml}

if [ ! -f "${ROBOT_CONFIG_FILE}" ]; then
    echo "Robot config file not found: ${ROBOT_CONFIG_FILE}" >&2
    exit 1
fi

# Set AMENT_PREFIX_PATH before sourcing - this is the key for package discovery
export AMENT_PREFIX_PATH=/robros/ws/install:/opt/ros/jazzy

source /opt/ros/jazzy/setup.bash
source /robros/ws/install/setup.bash

ulimit -r 90
ulimit -l unlimited

cd /robros/ws
exec ./manager \
    --config-file "${ROBOT_CONFIG_FILE}"
