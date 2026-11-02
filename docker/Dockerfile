# Online scene graph + ROS 2 Humble. Python 3.10 is Humble's interpreter, so rclpy,
# torch, record3d, and YOLO share one process — no conda/unix-socket split.
#
# GPU: NVIDIA driver on the host + nvidia-container-toolkit. PyTorch wheels
# ship the CUDA 11.8 runtime; this image does not install a full CUDA toolkit.
ARG BASE_IMAGE=osrf/ros:humble-desktop-full
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
    ROS_DISTRO=humble

# Open3D / OpenCV / Record3D USB. CycloneDDS matches the robot RMW.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-venv \
        python3-dev \
        python3-numpy \
        git \
        wget \
        libgl1 \
        libglib2.0-0 \
        libusb-1.0-0 \
        libusb-1.0-0-dev \
        usbutils \
        ros-humble-rmw-cyclonedds-cpp \
        && rm -rf /var/lib/apt/lists/*

# Match the conda stack: torch 2.0.1 + CUDA 11.8. Host driver must be recent enough.
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --no-cache-dir \
        torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
        --index-url https://download.pytorch.org/whl/cu118

COPY docker/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# Optional: 3D IoU helpers. Skip quietly if the wheel is gone.
RUN python3 -m pip install --no-cache-dir --no-deps pytorch3d \
        -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt201/download.html \
    || echo "WARN: pytorch3d wheel not installed; some IoU paths need it"

WORKDIR /ws
COPY setup.py /ws/setup.py
COPY online_scene_graph /ws/online_scene_graph
RUN python3 -m pip install --no-cache-dir -e /ws

# Find.srv — same Python as the mapper, so the service node can import rclpy in-process later
COPY ros/online_scene_graph_interfaces /opt/osg_ws/src/online_scene_graph_interfaces
WORKDIR /opt/osg_ws
RUN . /opt/ros/humble/setup.sh && colcon build --merge-install --packages-select online_scene_graph_interfaces \
    && rm -rf /opt/osg_ws/build /opt/osg_ws/log

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

COPY docker/cyclonedds_profile.xml /opt/cyclonedds_profile.xml
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    ROS_DOMAIN_ID=96 \
    CYCLONEDDS_URI=file:///opt/cyclonedds_profile.xml

WORKDIR /ws
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
