# IGRIS-C 시뮬레이션 스택 실행 환경
#
# 호스트(Ubuntu 22.04 / glibc 2.35)에서는 NUC 바이너리(GLIBC_2.38 요구)를 직접 못 돌린다.
# 컨테이너는 자기 glibc를 쓰므로 24.04 이미지로 우회한다.
# 부수 효과로 /dev/shm이 격리되어 실기 스택과의 SHM 충돌도 없다.
FROM ros:jazzy-ros-base

ARG ONNXRUNTIME_VERSION=1.20.1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git wget ca-certificates gdb \
      libeigen3-dev libboost-dev \
      libfmt9 libyaml-cpp0.8 libglfw3 libopengl0 libglx0 libgl1 \
      libncurses6 libtinfo6 libssl3t64 \
      ros-jazzy-xacro ros-jazzy-urdfdom ros-jazzy-rmw-cyclonedds-cpp \
      ros-jazzy-sensor-msgs ros-jazzy-nav-msgs ros-jazzy-statistics-msgs \
      ros-jazzy-rosgraph-msgs ros-jazzy-ament-index-cpp \
      python3-colcon-common-extensions python3-yaml \
 && rm -rf /var/lib/apt/lists/*

# ---- RBDL 3.3.1 (+ urdfreader addon) --------------------------------
# 바이너리가 librbdl.so.3.3.1 / librbdl_urdfreader.so.3.3.1 을 요구한다. apt에 없어서 소스 빌드.
# --recursive 필수 (urdfreader thirdparty 서브모듈), Boost 필요.
RUN . /opt/ros/jazzy/setup.sh \
 && git clone --depth 1 --branch v3.3.1 --recursive https://github.com/rbdl/rbdl.git /tmp/rbdl \
 && cmake -S /tmp/rbdl -B /tmp/rbdl/build \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DRBDL_BUILD_ADDON_URDFREADER=ON \
 && cmake --build /tmp/rbdl/build -j"$(nproc)" \
 && cmake --install /tmp/rbdl/build \
 && rm -rf /tmp/rbdl && ldconfig

# ---- ONNX Runtime ----------------------------------------------------
# controller(sim)와 igris_c_gmt_public 둘 다 libonnxruntime.so.1 을 쓴다.
RUN wget -q "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz" -O /tmp/ort.tgz \
 && tar xzf /tmp/ort.tgz -C /tmp \
 && cp -a /tmp/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}/lib/.     /usr/local/lib/ \
 && cp -a /tmp/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}/include/. /usr/local/include/ \
 && rm -rf /tmp/ort.tgz "/tmp/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}" \
 && ORT_REAL="$(cd /usr/local/lib && ls libonnxruntime.so.1.* | head -1)" \
 && ln -sf "/usr/local/lib/$ORT_REAL" /usr/lib/x86_64-linux-gnu/libonnxruntime.so.1 \
 && ldconfig
# 주의: 배포 tarball의 SONAME이 libonnxruntime.so.1.17.3 이라 ldconfig가 .so.1 을 등록하지 않는다.
# 바이너리는 NEEDED libonnxruntime.so.1 을 찾으므로, 로더가 파일명으로 뒤지는
# 기본 경로(/usr/lib/x86_64-linux-gnu)에 그 이름으로 심볼릭 링크를 둔다.
ENV LD_LIBRARY_PATH=/usr/local/lib

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
WORKDIR /igris
CMD ["bash"]

# ---- OpenGL ----------------------------------------------------------
# MuJoCo 뷰어가 GLX를 쓴다. 1순위는 --gpus all 로 넘겨준 NVIDIA 드라이버,
# 실패 시 mesa llvmpipe 소프트웨어 렌더링으로 떨어지게 DRI를 깔아둔다.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1-mesa-dri libglx-mesa0 libegl1 libxrandr2 libxinerama1 libxcursor1 libxi6 \
 && rm -rf /var/lib/apt/lists/*
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

# ---- 텔레옵 경로 ------------------------------------------------------
# GMR -> Redis -> redis_to_ros2_bridge.py -> /gmr/teleop/retarget_frame -> gmt_public
# 브릿지는 redis-py 없이 raw RESP 소켓을 쓰므로 서버만 있으면 된다.
# onnxruntime(python)은 헤드셋 없이 텔레옵 경로를 검증하는 gmr_frame_pub.py 용.
RUN apt-get update && apt-get install -y --no-install-recommends redis-server python3-pip \
 && rm -rf /var/lib/apt/lists/* \
 && pip3 install --no-cache-dir --break-system-packages onnxruntime==1.20.1

# ---- 재생/렌더 도구 -----------------------------------------------------
# state_recorder / render_replay 용. mujoco는 3.2.7 고정(상위 버전은 이 MJCF 거부).
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg xvfb \
 && rm -rf /var/lib/apt/lists/* \
 && pip3 install --no-cache-dir --break-system-packages mujoco==3.2.7

# ---- cyclonedds-python (S2 SONIC command-path: Python -> lowcmd) -------------
# Task-5 spike: a Python node publishes igris_c::msg::dds::LowCmd to the native
# topic igris_c_001/rt/lowcmd (best_effort), driving the SDK controller directly.
#
# VERSION IS LOAD-BEARING: it MUST match the SDK/ROS CycloneDDS 0.10.5. A newer
# wheel (e.g. cyclonedds 11.x) advertises an XTypes type object that the 0.10.5
# rmw_cyclonedds cannot parse ("ddsi_xt_type_init_impl with invalid type object")
# and SEGFAULTS the ROS nodes (igris_c_gmt_public) on discovery. So build the
# 0.10.5 python binding against this image's own ROS jazzy CycloneDDS 0.10.5.
#
# The IDL types are hand-written to match igris_c_msgs.hpp exactly (@final +
# @autoid("sequential"); Header.frame_id is char[256]; kinematic_modes is the
# 32-bit enum KinematicMode) — see tools/sonic/lowcmd_pub.py.
#
# FALLBACK if cyclonedds-python ever cannot publish an accepted lowcmd: add a C++
# policy_source:external mode to igris_c_gmt_public (plain ROS2 topic
# /sonic/policy_action -> action_builder -> lowcmd) and route the tracker through it.
# Build the binding against a synthetic prefix that points at ROS jazzy's own
# CycloneDDS 0.10.5 (headers + libddsc + libcycloneddsidl + idlc + cmake). At
# runtime libddsc.so.0 is found via the sourced ROS setup (LD_LIBRARY_PATH).
RUN mkdir -p /opt/cyclonedds/bin /opt/cyclonedds/lib \
 && ln -s /opt/ros/jazzy/include/CycloneDDS /opt/cyclonedds/include \
 && ln -s /opt/ros/jazzy/bin/idlc /opt/cyclonedds/bin/idlc \
 && ln -sf /opt/ros/jazzy/lib/x86_64-linux-gnu/libddsc.so*         /opt/cyclonedds/lib/ \
 && ln -sf /opt/ros/jazzy/lib/x86_64-linux-gnu/libcycloneddsidl.so* /opt/cyclonedds/lib/ \
 && ln -s /opt/ros/jazzy/lib/x86_64-linux-gnu/cmake/CycloneDDS      /opt/cyclonedds/lib/cmake \
 && CYCLONEDDS_HOME=/opt/cyclonedds \
    pip3 install --no-cache-dir --break-system-packages "cyclonedds==0.10.5"
