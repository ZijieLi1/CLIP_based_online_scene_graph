# shellcheck shell=bash
source /opt/ros/humble/setup.bash
if [[ -f /opt/osg_ws/install/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/osg_ws/install/setup.bash
fi
if [[ -f /ws/ros/install/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /ws/ros/install/setup.bash
fi
export PYTHONPATH="/ws:${PYTHONPATH:-}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-96}"
if [[ -z "${CYCLONEDDS_URI:-}" ]]; then
  if [[ -f /ws/docker/cyclonedds_profile.xml ]]; then
    export CYCLONEDDS_URI=file:///ws/docker/cyclonedds_profile.xml
  else
    export CYCLONEDDS_URI=file:///opt/cyclonedds_profile.xml
  fi
fi
