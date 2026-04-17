#!/usr/bin/env python3
"""Task 1: Kinematics-based pick and place in PyBullet.

This script uses inverse kinematics (IK) and Cartesian waypoint control
for a Franka Panda arm to pick a cube and place it at a target pose.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict

import numpy as np
import pybullet as p
import pybullet_data

ARM_JOINTS = list(range(7))
FINGER_JOINTS = [9, 10]
EE_LINK_INDEX = 11
SIM_TIMESTEP = 1.0 / 240.0


@dataclass
class PickPlaceResult:
    task_name: str
    pick_success: bool
    place_success: bool
    final_cube_position: list[float]
    target_place_position: list[float]
    position_error_l2: float
    position_error_xy: float
    position_error_z: float
    ee_at_pick_error: float


class PandaPickPlace:
    def __init__(self, gui: bool = False, realtime: bool = False, speed: str = "normal"):
        self.gui = gui
        self.realtime = realtime
        self.speed = speed
        self.client_id = p.connect(p.GUI if gui else p.DIRECT)
        if gui and hasattr(p, "COV_ENABLE_KEYBOARD_SHORTCUTS"):
            p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(SIM_TIMESTEP)

        self.plane_id = p.loadURDF("plane.urdf")
        self.table_id = p.loadURDF("table/table.urdf", basePosition=[0.5, 0.0, -0.65], useFixedBase=True)
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", basePosition=[0.0, 0.0, 0.0], useFixedBase=True)

        self.cube_size = 0.06
        cube_half = self.cube_size / 2.0
        cube_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[cube_half, cube_half, cube_half])
        cube_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[cube_half, cube_half, cube_half],
            rgbaColor=[0.95, 0.45, 0.12, 1.0],
        )
        self.cube_id = p.createMultiBody(
            baseMass=0.10,
            baseCollisionShapeIndex=cube_collision,
            baseVisualShapeIndex=cube_visual,
            basePosition=[0.58, 0.00, cube_half],
            baseOrientation=[0, 0, 0, 1],
        )
        self.attachment_constraint = None
        self._last_ee_pos = None
        self._ever_attached = False

        if speed == "normal":
            self.step_scale = 1.0
            self.arm_position_gain = 0.07
            self.teleop_position_gain = 0.08
        elif speed == "fast":
            self.step_scale = 0.55
            self.arm_position_gain = 0.10
            self.teleop_position_gain = 0.10
        elif speed == "ultrafast":
            self.step_scale = 0.30
            self.arm_position_gain = 0.14
            self.teleop_position_gain = 0.12
        else:
            raise ValueError(f"Unsupported speed preset: {speed}")

        self._reset_robot()
        self._draw_static_reference_frames()

    def _scaled_steps(self, steps: int, min_steps: int = 8) -> int:
        return max(min_steps, int(round(steps * self.step_scale)))

    def _step(self, steps: int) -> None:
        for _ in range(steps):
            p.stepSimulation()
            self._update_ee_trace()
            if self.gui and self.realtime:
                time.sleep(SIM_TIMESTEP)

    def _draw_frame(self, position, orientation, axis_len: float = 0.08, label: str | None = None) -> None:
        if not self.gui:
            return

        rot = p.getMatrixFromQuaternion(orientation)
        x_axis = [rot[0], rot[3], rot[6]]
        y_axis = [rot[1], rot[4], rot[7]]
        z_axis = [rot[2], rot[5], rot[8]]

        def end_point(axis):
            return [position[i] + axis_len * axis[i] for i in range(3)]

        p.addUserDebugLine(position, end_point(x_axis), [1, 0, 0], lineWidth=2, lifeTime=0)
        p.addUserDebugLine(position, end_point(y_axis), [0, 1, 0], lineWidth=2, lifeTime=0)
        p.addUserDebugLine(position, end_point(z_axis), [0, 0, 1], lineWidth=2, lifeTime=0)
        if label is not None:
            p.addUserDebugText(label, [position[0], position[1], position[2] + axis_len * 1.15], [0, 0, 0], textSize=1.2)

    def _draw_static_reference_frames(self) -> None:
        if not self.gui:
            return

        self._draw_frame([0.0, 0.0, 0.0], [0, 0, 0, 1], axis_len=0.12, label="world")

        cube_pos, cube_orn = p.getBasePositionAndOrientation(self.cube_id)
        self._draw_frame(cube_pos, cube_orn, axis_len=0.06, label="cube_init")

        place_target = [0.42, -0.22, self.cube_size / 2.0]
        place_orn = p.getQuaternionFromEuler([math.pi, 0.0, -math.pi / 2.0])
        self._draw_frame(place_target, place_orn, axis_len=0.06, label="place_target")
        p.addUserDebugText("target_cube", [cube_pos[0], cube_pos[1], cube_pos[2] + 0.08], [0.05, 0.05, 0.05], textSize=1.2)

    def _draw_waypoint(self, position, yaw_rad: float, label: str) -> None:
        if not self.gui:
            return
        orn = p.getQuaternionFromEuler([math.pi, 0.0, yaw_rad])
        self._draw_frame(position, orn, axis_len=0.05, label=label)

    def _update_ee_trace(self) -> None:
        if not self.gui:
            return

        ee_state = p.getLinkState(self.robot_id, EE_LINK_INDEX, computeForwardKinematics=True)
        ee_pos = ee_state[4]
        if self._last_ee_pos is not None:
            p.addUserDebugLine(self._last_ee_pos, ee_pos, [1.0, 0.2, 0.2], lineWidth=1.5, lifeTime=0)
        self._last_ee_pos = ee_pos

    def _reset_robot(self) -> None:
        home = [0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.8]
        for j_idx, angle in zip(ARM_JOINTS, home):
            p.resetJointState(self.robot_id, j_idx, angle)
        self.set_gripper(open_width=0.08, settle_steps=self._scaled_steps(80, min_steps=20))
        self._step(self._scaled_steps(120, min_steps=30))

    def _drive_gripper_once(self, open_width: float) -> None:
        finger_joint_target = max(0.0, min(0.04, open_width / 2.0))
        for j_idx in FINGER_JOINTS:
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=j_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=finger_joint_target,
                force=40,
            )

    def set_gripper(self, open_width: float, settle_steps: int = 120) -> None:
        for _ in range(settle_steps):
            self._drive_gripper_once(open_width)
            p.stepSimulation()
            self._update_ee_trace()
            if self.gui and self.realtime:
                time.sleep(SIM_TIMESTEP)

    def move_ee_pose(self, position: list[float], yaw_rad: float = 0.0, steps: int = 240) -> None:
        # Gripper faces downward: roll=pi, pitch=0, yaw as requested.
        target_orn = p.getQuaternionFromEuler([math.pi, 0.0, yaw_rad])

        joint_targets = p.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=EE_LINK_INDEX,
            targetPosition=position,
            targetOrientation=target_orn,
            maxNumIterations=200,
            residualThreshold=1e-4,
        )

        for _ in range(steps):
            for j_local, joint_id in enumerate(ARM_JOINTS):
                p.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=joint_id,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=joint_targets[j_local],
                    force=120,
                    positionGain=self.arm_position_gain,
                    velocityGain=1.0,
                )
            p.stepSimulation()
            self._update_ee_trace()
            if self.gui and self.realtime:
                time.sleep(SIM_TIMESTEP)

    def attach_cube_to_ee(self) -> float:
        ee_state = p.getLinkState(self.robot_id, EE_LINK_INDEX, computeForwardKinematics=True)
        ee_pos, ee_orn = ee_state[4], ee_state[5]

        cube_pos, cube_orn = p.getBasePositionAndOrientation(self.cube_id)

        inv_ee_pos, inv_ee_orn = p.invertTransform(ee_pos, ee_orn)
        rel_pos, rel_orn = p.multiplyTransforms(inv_ee_pos, inv_ee_orn, cube_pos, cube_orn)

        self.attachment_constraint = p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=EE_LINK_INDEX,
            childBodyUniqueId=self.cube_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=rel_pos,
            parentFrameOrientation=rel_orn,
            childFramePosition=[0, 0, 0],
            childFrameOrientation=[0, 0, 0, 1],
        )
        self._ever_attached = True

        return math.dist(list(ee_pos), list(cube_pos))

    def release_cube(self) -> None:
        if self.attachment_constraint is not None:
            p.removeConstraint(self.attachment_constraint)
            self.attachment_constraint = None

    def _maybe_attach_for_teleop(self, gripper_width: float) -> None:
        if self.attachment_constraint is not None:
            if gripper_width > 0.045:
                self.release_cube()
            return

        if gripper_width > 0.025:
            return

        ee_state = p.getLinkState(self.robot_id, EE_LINK_INDEX, computeForwardKinematics=True)
        ee_pos = ee_state[4]
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        dist = math.dist(list(ee_pos), list(cube_pos))
        if dist < 0.075:
            self.attach_cube_to_ee()

    def interactive_loop(self, mode: str = "both") -> None:
        if not self.gui:
            raise ValueError("Interactive mode requires --gui")

        ee_state = p.getLinkState(self.robot_id, EE_LINK_INDEX, computeForwardKinematics=True)
        ee_pos = list(ee_state[4])
        yaw = 0.0
        gripper_width = 0.08
        pos_step = 0.004
        yaw_step = 0.04
        grip_step = 0.004

        slider_ids = {}
        last_slider_vals = None
        if mode in ("slider", "both"):
            slider_ids = {
                "x": p.addUserDebugParameter("ee_x", 0.35, 0.75, ee_pos[0]),
                "y": p.addUserDebugParameter("ee_y", -0.35, 0.35, ee_pos[1]),
                "z": p.addUserDebugParameter("ee_z", 0.02, 0.45, ee_pos[2]),
                "yaw": p.addUserDebugParameter("ee_yaw", -math.pi, math.pi, yaw),
                "gripper": p.addUserDebugParameter("gripper", 0.0, 0.08, gripper_width),
            }
            last_slider_vals = {name: p.readUserDebugParameter(uid) for name, uid in slider_ids.items()}

        p.addUserDebugText(
            "Controls: WASD/QE move, J/L yaw, O/P gripper, ESC quit",
            [0.25, -0.55, 0.55],
            [0.1, 0.1, 0.1],
            textSize=1.2,
        )

        exit_reason = "window_disconnected"
        while p.isConnected(self.client_id):
            if mode in ("slider", "both"):
                try:
                    slider_vals = {name: p.readUserDebugParameter(uid) for name, uid in slider_ids.items()}
                except p.error:
                    exit_reason = "slider_disconnected"
                    break
                changed = any(abs(slider_vals[name] - last_slider_vals[name]) > 1e-6 for name in slider_vals)
                if changed:
                    ee_pos = [slider_vals["x"], slider_vals["y"], slider_vals["z"]]
                    yaw = slider_vals["yaw"]
                    gripper_width = slider_vals["gripper"]
                    last_slider_vals = slider_vals

            if mode in ("keyboard", "both"):
                try:
                    events = p.getKeyboardEvents()
                except p.error:
                    exit_reason = "keyboard_disconnected"
                    break
                if ord("w") in events and events[ord("w")] & p.KEY_IS_DOWN:
                    ee_pos[0] += pos_step
                if ord("s") in events and events[ord("s")] & p.KEY_IS_DOWN:
                    ee_pos[0] -= pos_step
                if ord("a") in events and events[ord("a")] & p.KEY_IS_DOWN:
                    ee_pos[1] += pos_step
                if ord("d") in events and events[ord("d")] & p.KEY_IS_DOWN:
                    ee_pos[1] -= pos_step
                if ord("q") in events and events[ord("q")] & p.KEY_IS_DOWN:
                    ee_pos[2] += pos_step
                if ord("e") in events and events[ord("e")] & p.KEY_IS_DOWN:
                    ee_pos[2] -= pos_step
                if ord("j") in events and events[ord("j")] & p.KEY_IS_DOWN:
                    yaw += yaw_step
                if ord("l") in events and events[ord("l")] & p.KEY_IS_DOWN:
                    yaw -= yaw_step
                if ord("o") in events and events[ord("o")] & p.KEY_IS_DOWN:
                    gripper_width = max(0.0, gripper_width - grip_step)
                if ord("p") in events and events[ord("p")] & p.KEY_IS_DOWN:
                    gripper_width = min(0.08, gripper_width + grip_step)
                if p.B3G_ESCAPE in events and events[p.B3G_ESCAPE] & p.KEY_WAS_TRIGGERED:
                    exit_reason = "esc_pressed"
                    break

            ee_pos[0] = float(np.clip(ee_pos[0], 0.35, 0.75))
            ee_pos[1] = float(np.clip(ee_pos[1], -0.35, 0.35))
            ee_pos[2] = float(np.clip(ee_pos[2], 0.02, 0.45))
            gripper_width = float(np.clip(gripper_width, 0.0, 0.08))

            target_orn = p.getQuaternionFromEuler([math.pi, 0.0, yaw])
            joint_targets = p.calculateInverseKinematics(
                bodyUniqueId=self.robot_id,
                endEffectorLinkIndex=EE_LINK_INDEX,
                targetPosition=ee_pos,
                targetOrientation=target_orn,
                maxNumIterations=120,
                residualThreshold=1e-4,
            )
            for j_local, joint_id in enumerate(ARM_JOINTS):
                p.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=joint_id,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=joint_targets[j_local],
                    force=120,
                    positionGain=self.teleop_position_gain,
                    velocityGain=1.0,
                )
            self._drive_gripper_once(gripper_width)
            self._maybe_attach_for_teleop(gripper_width)
            p.stepSimulation()
            self._update_ee_trace()
            if self.realtime:
                time.sleep(SIM_TIMESTEP)
        print(f"[interactive] exit_reason={exit_reason}")

    def interactive_summary(self) -> dict:
        if not p.isConnected(self.client_id):
            return {
                "task_name": "Task1-Kinematics-PyBullet-Interactive",
                "status": "gui_disconnected_before_summary",
                "ever_attached": self._ever_attached,
            }
        ee_state = p.getLinkState(self.robot_id, EE_LINK_INDEX, computeForwardKinematics=True)
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        return {
            "task_name": "Task1-Kinematics-PyBullet-Interactive",
            "final_ee_position": list(ee_state[4]),
            "final_cube_position": list(cube_pos),
            "cube_attached": self.attachment_constraint is not None,
            "ever_attached": self._ever_attached,
        }

    def run_pick_place(self) -> PickPlaceResult:
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)

        pre_pick = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.16]
        pick_pose = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.045]

        place_target = [0.42, -0.22, self.cube_size / 2.0]
        pre_place = [place_target[0], place_target[1], place_target[2] + 0.16]
        place_pose = [place_target[0], place_target[1], place_target[2] + 0.05]

        self._draw_waypoint(pre_pick, 0.0, "pre_pick")
        self._draw_waypoint(pick_pose, 0.0, "pick")
        self._draw_waypoint(pre_place, -math.pi / 2.0, "pre_place")
        self._draw_waypoint(place_pose, -math.pi / 2.0, "place")

        self.set_gripper(open_width=0.08, settle_steps=self._scaled_steps(100, min_steps=20))
        self.move_ee_pose(pre_pick, yaw_rad=0.0, steps=self._scaled_steps(280, min_steps=40))
        self.move_ee_pose(pick_pose, yaw_rad=0.0, steps=self._scaled_steps(220, min_steps=35))

        ee_to_cube_before = self.attach_cube_to_ee()
        pick_success = ee_to_cube_before < 0.08

        self.set_gripper(open_width=0.00, settle_steps=self._scaled_steps(160, min_steps=24))
        self.move_ee_pose(pre_pick, yaw_rad=0.0, steps=self._scaled_steps(220, min_steps=35))
        self.move_ee_pose(pre_place, yaw_rad=-math.pi / 2.0, steps=self._scaled_steps(320, min_steps=50))
        self.move_ee_pose(place_pose, yaw_rad=-math.pi / 2.0, steps=self._scaled_steps(220, min_steps=35))

        self.release_cube()
        self.set_gripper(open_width=0.08, settle_steps=self._scaled_steps(120, min_steps=20))
        self.move_ee_pose(pre_place, yaw_rad=-math.pi / 2.0, steps=self._scaled_steps(220, min_steps=35))

        # Let the cube settle after release.
        self._step(self._scaled_steps(240, min_steps=40))

        final_cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)

        dx = final_cube_pos[0] - place_target[0]
        dy = final_cube_pos[1] - place_target[1]
        dz = final_cube_pos[2] - place_target[2]

        error_l2 = math.sqrt(dx * dx + dy * dy + dz * dz)
        error_xy = math.sqrt(dx * dx + dy * dy)
        error_z = abs(dz)

        place_success = error_xy < 0.05 and error_z < 0.05

        return PickPlaceResult(
            task_name="Task1-Kinematics-PyBullet-PickPlace",
            pick_success=pick_success,
            place_success=place_success,
            final_cube_position=list(final_cube_pos),
            target_place_position=place_target,
            position_error_l2=error_l2,
            position_error_xy=error_xy,
            position_error_z=error_z,
            ee_at_pick_error=ee_to_cube_before,
        )

    def close(self) -> None:
        if p.isConnected(self.client_id):
            p.disconnect(self.client_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kinematics-based pick and place in PyBullet")
    parser.add_argument("--gui", action="store_true", help="Run with PyBullet GUI")
    parser.add_argument("--realtime", action="store_true", help="Slow down simulation to visualize the motion")
    parser.add_argument(
        "--speed",
        choices=["normal", "fast", "ultrafast"],
        default="normal",
        help="Preset for automatic motion speed",
    )
    parser.add_argument(
        "--interactive",
        choices=["none", "keyboard", "slider", "both"],
        default="none",
        help="Enable manual teleoperation with keyboard, sliders, or both",
    )
    parser.add_argument(
        "--output",
        default="results/task1_pybullet_result.json",
        help="Path to write JSON result",
    )
    args = parser.parse_args()

    realtime = args.realtime or args.interactive != "none"
    sim = PandaPickPlace(gui=args.gui, realtime=realtime, speed=args.speed)
    try:
        if args.interactive == "none":
            result = sim.run_pick_place()
            result_dict = asdict(result)
        else:
            sim.interactive_loop(mode=args.interactive)
            result_dict = sim.interactive_summary()
    finally:
        sim.close()

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)

    print(json.dumps(result_dict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
