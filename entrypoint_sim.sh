#!/usr/bin/env bash
# ==============================================================================
# Sorter Simulation Service Script (runs inside the 'sim' Docker container)
#   start | stop | restart | status | foreground
# ==============================================================================
set -e

CONTAINER_WS="/root/ros_ws"
FAST_DDS_XML="${CONTAINER_WS}/fastdds_udp.xml"
WORLD_SDF="${CONTAINER_WS}/worlds/sorter.sdf"
GROUNDTRUTH="${CONTAINER_WS}/worlds/groundtruth.json"

export IGN_IP=127.0.0.1
export FASTRTPS_DEFAULT_PROFILES_FILE="${FAST_DDS_XML}"

source /opt/ros/humble/setup.bash

# GPU rendering when the container was started with --gpus (NVIDIA EGL, no X needed);
# otherwise fall back to llvmpipe software rendering on an Xvfb display.
if nvidia-smi -L > /dev/null 2>&1; then
    RENDER_MODE="gpu"
    unset LIBGL_ALWAYS_SOFTWARE
    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
else
    RENDER_MODE="software"
    export DISPLAY=:99
    export LIBGL_ALWAYS_SOFTWARE=1
fi

stop_services() {
    echo "[sim] Stopping simulation services..."
    # SIGKILL: Gazebo intermittently ignores SIGTERM mid-physics-step, leaving
    # a stale world behind. All processes here are stateless.
    pkill -9 -f "ros_gz_bridge" || true
    pkill -9 -f "ign gazebo" || true
    pkill -9 -f "gz_sim.launch.py" || true
    pkill -9 -f "ign-transport-topic" || true
    pkill -9 -f "Xvfb" || true
    echo "[sim] Simulation stopped."
}

# Bridge argument list: fixed robot/camera topics + one attach/detach pair per
# block listed in groundtruth.json (written by randomize_world.py).
bridge_args() {
    echo "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"
    echo "/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image"
    echo "/camera/perspective/image_raw@sensor_msgs/msg/Image[gz.msgs.Image"
    echo "/camera/front/image_raw@sensor_msgs/msg/Image[gz.msgs.Image"
    echo "/sorter_robot/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model"
    echo "/world/sorter/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V"
    for j in 1 2 3 4; do
        echo "/model/sorter_robot/joint/joint_${j}/cmd_pos@std_msgs/msg/Float64]gz.msgs.Double"
    done
    python3 -c "
import json
for k in json.load(open('${GROUNDTRUTH}')):
    print(f'/electromagnet/attach_{k}@std_msgs/msg/Empty]gz.msgs.Empty')
    print(f'/electromagnet/detach_{k}@std_msgs/msg/Empty]gz.msgs.Empty')
"
}

start_services() {
    echo "[sim] Initializing simulation stack (rendering: ${RENDER_MODE})..."

    # DetachableJoint build with <initial_attach>false</initial_attach> support
    if [ -f "${CONTAINER_WS}/plugins/libignition-gazebo-detachable-joint-system.so" ]; then
        cp "${CONTAINER_WS}/plugins/libignition-gazebo-detachable-joint-system.so" /usr/lib/x86_64-linux-gnu/ign-gazebo-6/plugins/ 2>/dev/null || true
    fi

    if [ "${RENDER_MODE}" = "software" ] && ! pgrep -f "Xvfb :99" > /dev/null; then
        echo "[sim] Starting Xvfb on display :99..."
        Xvfb :99 -screen 0 1024x768x24 > /tmp/xvfb.log 2>&1 &
        sleep 1
    fi

    if ! pgrep -f "ign gazebo" > /dev/null; then
        echo "[sim] Launching Gazebo Fortress (headless)..."
        ign gazebo -s -r "${WORLD_SDF}" --headless-rendering > /tmp/sorter.log 2>&1 &
        echo -n "[sim] Waiting for Gazebo transport"
        for _ in {1..40}; do
            if timeout 1 ign topic -e -t /clock -n 1 > /dev/null 2>&1; then
                echo " [Ready]"
                break
            fi
            echo -n "."
            sleep 0.5
        done
    else
        echo "[sim] Gazebo already running (use 'restart' to load a regenerated world)."
    fi

    if ! pgrep -f "ros_gz_bridge parameter_bridge" > /dev/null; then
        echo "[sim] Starting ros_gz_bridge..."
        mapfile -t ARGS < <(bridge_args)
        ros2 run ros_gz_bridge parameter_bridge "${ARGS[@]}" \
            --ros-args -r /sorter_robot/joint_states:=/joint_states \
                       -r /world/sorter/dynamic_pose/info:=/sim/poses \
            > /tmp/bridges.log 2>&1 &
        sleep 1
        echo "[sim] Bridge started (${#ARGS[@]} topics)."
    else
        echo "[sim] ros_gz_bridge already running."
    fi
}

trap 'stop_services; exit 0' SIGINT SIGTERM

case "$1" in
    stop)
        stop_services
        ;;
    restart)
        stop_services
        sleep 0.5
        start_services
        ;;
    status)
        echo "=== Simulation Process Status (rendering: ${RENDER_MODE}) ==="
        pgrep -a -f "Xvfb :99" || echo "Xvfb: not running (not needed for GPU rendering)"
        pgrep -a -f "ign gazebo" || echo "Gazebo: not running"
        pgrep -a -f "ros_gz_bridge parameter_bridge" | cut -c1-120 || echo "Bridge: not running"
        ;;
    foreground)
        start_services
        echo "[sim] Running in foreground. Press Ctrl+C to stop."
        wait -n || true
        ;;
    start|*)
        start_services
        ;;
esac
