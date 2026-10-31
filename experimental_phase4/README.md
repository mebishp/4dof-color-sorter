# Experimental Phase-4 work (archived, not active)

Archived during the revert-to-demo pass. The active demo uses open-loop
`JointPositionController` servos + `pick_and_place.py` on the host.

Contents:
- `controllers.yaml` — `joint_state_broadcaster` + `arm_controller`
  (`JointTrajectoryController`) for `ign_ros2_control`.
- `sorter_control.urdf` — full `sorter_robot.urdf` + `magnetic_tool_link` +
  `<ros2_control>` (`ign_ros2_control/IgnitionSystem`, position interface).
  `robot_state_publisher` must serve this as `/robot_description` for the
  Gazebo `ign_ros2_control` plugin to initialize `controller_manager`.
- `test_physics_stability.py` — GZ pose bounds/teleport + CSV vocabulary test.
  Register under `SUITES['e2e']` in `test_runner.py` to re-enable.

To resume Phase-4 you also need (were reverted, see `AGENT_HANDOFF.md`):
1. `ign_ros2_control` plugin in `worlds/sorter.sdf.template` (replacing the
   `JointPositionController` blocks) + joint cmd_pos bridge removal in
   `entrypoint_sim.sh` + RSP/spawner startup there.
2. `pick_and_place.py` rewritten around the
   `/arm_controller/follow_joint_trajectory` action. NOTE: host Arch ROS lacks
   `control_msgs`, so the sort must run inside the container; also
   top-down-camera grasp verification must be post-transit (a held block is
   still visible at the pick XY — post-lift "disappearance" checks
   false-trigger and their DETACH drops good grasps).
3. Sim real-time factor is ~0.27–0.59 (software rendering); camera
   `update_rate` reduction to 10 Hz roughly doubled it.

Container image `sim:with-tools` already contains
`ros-humble-ign-ros2-control`, `joint-trajectory-controller`,
`joint-state-broadcaster`, `controller-manager`, `ros2controlcli`.
