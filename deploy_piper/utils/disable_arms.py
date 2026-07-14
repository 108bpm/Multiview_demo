#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Gently disable (软失能) a Piper arm so it does NOT free-fall.

Sequence: (1) optionally move the arm slowly to home (all joints 0) with position
control, then (2) softly release it via MIT control.

Usage:
    python utils/gentle_disable_arm.py                       # all 4 arms (dual-arm)
    python utils/gentle_disable_arm.py left_follower
    python utils/gentle_disable_arm.py left_follower right_follower
    python utils/gentle_disable_arm.py left_follower --duration 3.0 --kp 12 --kd 1.2
    python utils/gentle_disable_arm.py left_follower --no-home   # skip move-to-home

Tuning on real hardware:
    --kp   initial holding gain (SDK reference 10). Higher = firmer hold at start.
    --kd   damping gain (SDK reference 0.8, max 5). Higher = slower/softer droop,
           too high = jitter/oscillation.
    --duration  seconds to ramp kp from --kp down to 0.
"""
import argparse
import time

from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW

NUM_JOINTS = 6
RATE_HZ = 100.0
HOME_TOL_RAD = 0.05  # ~3 deg: consider "at home" when every joint is within this


def make_robot(can):
    cfg = create_agx_arm_config(
        robot=ArmModel.PIPER,
        firmeware_version=PiperFW.V188,
        interface="socketcan",
        channel=can,
    )
    robot = AgxArmFactory.create_arm(cfg)
    gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
    robot.connect()
    return robot, gripper


def read_joints_rad(robot):
    ja = robot.get_joint_angles()
    return list(ja.msg) if ja is not None else [0.0] * NUM_JOINTS


def enable(robot, timeout=3.0):
    """Make sure motors are enabled so we can take MIT control before releasing."""
    start = time.time()
    while time.time() - start < timeout:
        if robot.enable(255):
            return True
        time.sleep(0.05)
    return False


def move_to_home(robot, gripper, speed, timeout=10.0):
    """Slowly move all joints to 0 with position control; wait until arrived."""
    robot.set_speed_percent(speed)
    start = time.time()
    while time.time() - start < timeout:
        if max(abs(a) for a in read_joints_rad(robot)) < HOME_TOL_RAD:
            return True
        robot.move_j([0.0] * NUM_JOINTS)
        gripper.move_gripper_m(0.0, 1.0)
        time.sleep(0.05)
    return False


def gentle_disable(can, kp0, kd, duration, go_home, home_speed, settle=0.6):
    robot, gripper = make_robot(can)
    time.sleep(0.2)

    if not enable(robot):
        print(f"[WARN] {can}: could not confirm enable — arm may already be limp.")

    if go_home:
        print(f"{can}: moving to home (speed {home_speed}%)...")
        if move_to_home(robot, gripper, home_speed):
            print(f"{can}: reached home.")
        else:
            print(f"[WARN] {can}: home move timed out — releasing from current pose.")

    hold = read_joints_rad(robot)
    print(f"{can}: holding {[round(a, 3) for a in hold]} rad, ramping kp {kp0}->0 over {duration}s")

    dt = 1.0 / RATE_HZ
    steps = max(1, int(duration * RATE_HZ))

    # Phase 1: ramp kp -> 0 while damping (kd) resists any fall.
    for s in range(steps + 1):
        kp = kp0 * (1.0 - s / steps)  # linear fade to zero
        for j in range(NUM_JOINTS):
            robot.move_mit(joint_index=j + 1, p_des=hold[j], v_des=0.0, kp=kp, kd=kd, t_ff=0.0)
        time.sleep(dt)

    # Phase 2: kp = 0, keep only damping so it settles softly at the bottom.
    for _ in range(int(settle * RATE_HZ)):
        for j in range(NUM_JOINTS):
            robot.move_mit(joint_index=j + 1, p_des=hold[j], v_des=0.0, kp=0.0, kd=kd, t_ff=0.0)
        time.sleep(dt)

    # Phase 3: fully release (next move_j restores position/speed control mode).
    while not robot.disable(255):
        time.sleep(0.01)
    time.sleep(0.3)
    print(f"{can}: 软失能成功!!!!")


def main():
    ap = argparse.ArgumentParser(description="Gently disable one or more Piper arms.")
    ap.add_argument("cans", nargs="*",
                    default=["left_leader", "left_follower", "right_leader", "right_follower"],
                    help="CAN names (default: all 4 arms). See can_arm_mapping.md")
    ap.add_argument("--kp", type=float, default=10.0, help="initial holding gain (ref 10)")
    ap.add_argument("--kd", type=float, default=0.8, help="damping gain (ref 0.8, max 5)")
    ap.add_argument("--duration", type=float, default=2.0, help="kp ramp-down seconds")
    ap.add_argument("--home-speed", type=int, default=15,
                    help="move-to-home speed percent (default 15, slow)")
    ap.add_argument("--no-home", dest="home", action="store_false",
                    help="skip move-to-home; soft-release from current pose")
    args = ap.parse_args()

    for can in args.cans:
        gentle_disable(can, args.kp, args.kd, args.duration, args.home, args.home_speed)


if __name__ == "__main__":
    main()
