import numpy as np
import pybullet as p
import time, threading, math

from .simulation import init_simulation, load_robot
from .kinematics import computeKinematicsAndJacobian, solveDLS
from .utils import quat_conjugate, quat_mul

class MiniArmVelocityController:
    def __init__(self):
        print("Initializing Mini Arm with improved velocity control...")

        # 控制參數
        self.joint_speed_max_ = 3.0  # rad/s
        self.linear_thresh_mm_ = 1.0  # mm
        self.angular_thresh_ = 0.01   # rad
        self.vel_gain_ = 1.5
        self.vel_gain_angular_ = 1.2
        self.ki_gain_ = 0.1
        self.lambda_dls_ = 0.05
        self.dt_ = 0.02

        # 狀態
        self.q_ = np.zeros(6, dtype=np.float32)
        self.current_position_ = np.zeros(3, dtype=np.float32)
        self.current_orientation_ = np.zeros(3, dtype=np.float32)
        self.J_ = np.zeros((6, 6), dtype=np.float32)
        self.error_integral_ = np.zeros(6, dtype=np.float32)
        self.max_integral_ = 10.0

        # 模擬與機器人
        self.physicsClient, self.planeId = init_simulation(gui=True)
        self.robotId = load_robot()

        # 關節與末端
        self.joint_indices = []
        self.joint_names = []
        for i in range(p.getNumJoints(self.robotId)):
            info = p.getJointInfo(self.robotId, i)
            if info[2] != p.JOINT_FIXED:
                self.joint_indices.append(i)
                self.joint_names.append(info[1].decode('utf-8'))
        if len(self.joint_indices) < 6:
            print("Warning: less than 6 actuated joints found. Found:", len(self.joint_indices))
        self.end_effector_index = self.joint_indices[-1] if self.joint_indices else p.getNumJoints(self.robotId) - 1
        print(f"End-effector link index: {self.end_effector_index}")
        print(f"Found {len(self.joint_indices)} controllable joints: {self.joint_names}")

        # 初始姿勢
        self._set_initial_pose()
        self._update_joint_states()

        # 模擬執行緒
        self.running = True
        self.sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.sim_thread.start()

        # 初始運動學
        self.computeKinematicsAndJacobian()
        print(f"Mini Arm initialized. Position: ({self.current_position_[0]:.1f}, {self.current_position_[1]:.1f}, {self.current_position_[2]:.1f}) mm")

    def _set_initial_pose(self):
        initial_pose = [0, 0, 0, 0, 0, 0]
        for i, joint_idx in enumerate(self.joint_indices[:6]):
            p.resetJointState(self.robotId, joint_idx, initial_pose[i] if i < len(initial_pose) else 0)
            p.setJointMotorControl2(self.robotId, joint_idx, p.VELOCITY_CONTROL, targetVelocity=0, force=500)

    def _update_joint_states(self):
        for i, joint_idx in enumerate(self.joint_indices[:6]):
            self.q_[i] = p.getJointState(self.robotId, joint_idx)[0]

    def _simulation_loop(self):
        while self.running:
            p.stepSimulation()
            time.sleep(1./240.)

    def computeKinematicsAndJacobian(self):
        pos, ori, J = computeKinematicsAndJacobian(self.robotId, self.end_effector_index, self.joint_indices, self.q_)
        if pos is not None:
            self.current_position_ = pos
            self.current_orientation_ = ori
            self.J_ = J

    def _stop_all_motors(self):
        for joint_idx in self.joint_indices[:6]:
            p.setJointMotorControl2(self.robotId, joint_idx, p.VELOCITY_CONTROL, targetVelocity=0, force=500)

    def stepToward(self, ox_deg, oy_deg, oz_deg, px, py, pz):
        """單步：PI 控制 + 四元數姿態誤差 + DLS"""
        target_o = np.array([np.deg2rad(ox_deg), np.deg2rad(oy_deg), np.deg2rad(oz_deg)], dtype=np.float32)
        target_p = np.array([px, py, pz], dtype=np.float32)

        # 工作空間檢查
        target_distance = np.linalg.norm(target_p)
        max_reach = 596.0
        if target_distance > max_reach:
            print(f"Warning: Target ({px:.1f}, {py:.1f}, {pz:.1f}) may be out of reach (distance: {target_distance:.1f}mm > {max_reach}mm)")
            scale = (max_reach - 10) / target_distance
            target_p = target_p * scale
            print(f"Adjusted target to: ({target_p[0]:.1f}, {target_p[1]:.1f}, {target_p[2]:.1f})")

        # 狀態更新
        self._update_joint_states()
        self.computeKinematicsAndJacobian()

        # 姿態誤差（四元數）
        q_target = p.getQuaternionFromEuler([float(target_o[0]), float(target_o[1]), float(target_o[2])])
        q_current = p.getQuaternionFromEuler([float(self.current_orientation_[0]),
                                              float(self.current_orientation_[1]),
                                              float(self.current_orientation_[2])])

        q_err = quat_mul(q_target, quat_conjugate(q_current))
        qw = max(min(q_err[3], 1.0), -1.0)
        angle = 2.0 * math.acos(qw)
        if abs(angle) < 1e-6:
            axis = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        else:
            s = math.sqrt(1.0 - qw*qw)
            if s < 1e-8:
                axis = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            else:
                axis = np.array([q_err[0]/s, q_err[1]/s, q_err[2]/s], dtype=np.float32)
        error_ang_vec = axis * angle

        # 位置誤差（mm）
        error_pos = target_p - self.current_position_

        # 臨界判斷
        ang_err_norm = np.linalg.norm(error_ang_vec)
        lin_err = np.linalg.norm(error_pos)
        if lin_err < self.linear_thresh_mm_ and ang_err_norm < self.angular_thresh_:
            self._stop_all_motors()
            self.error_integral_ = np.zeros(6)
            return True

        # 積分
        self.error_integral_[:3] += error_ang_vec * self.dt_
        self.error_integral_[3:] += error_pos * self.dt_
        self.error_integral_ = np.clip(self.error_integral_, -self.max_integral_, self.max_integral_)

        # PI
        w = error_ang_vec * self.vel_gain_angular_ + self.error_integral_[:3] * self.ki_gain_
        v = error_pos * self.vel_gain_ + self.error_integral_[3:] * self.ki_gain_

        # 距離自適應
        if lin_err > 100:
            v *= 1.5
        elif lin_err < 10:
            v *= 0.7

        # [角, 線] -> xdot
        xdot = np.concatenate([w, v]).astype(np.float32)

        # DLS 解 qdot
        qdot = solveDLS(self.J_, xdot, self.lambda_dls_)

        # 限速
        max_abs = np.max(np.abs(qdot))
        if max_abs > self.joint_speed_max_:
            scale_factor = self.joint_speed_max_ / max_abs
            qdot *= scale_factor
            self.error_integral_ *= 0.9

        # deadzone
        deadzone = 5e-4
        qdot[np.abs(qdot) < deadzone] = 0.0

        # 下發速度
        for i, joint_idx in enumerate(self.joint_indices[:6]):
            p.setJointMotorControl2(self.robotId, joint_idx, p.VELOCITY_CONTROL,
                                    targetVelocity=float(qdot[i]), force=1500)

        # 卡住檢測
        if not hasattr(self, 'stall_counter'):
            self.stall_counter = 0
        if not hasattr(self, 'last_position'):
            self.last_position = self.current_position_.copy()

        position_change = np.linalg.norm(self.current_position_ - self.last_position)
        if position_change < 0.1:
            self.stall_counter += 1
            if self.stall_counter > 50:
                self.error_integral_ *= 0.5
                self.stall_counter = 0
        else:
            self.stall_counter = 0

        self.last_position = self.current_position_.copy()
        return False

    def moveToPosition(self, target_x, target_y, target_z, target_ox=0, target_oy=0, target_oz=0, timeout=15.0):
        print(f"Moving to: pos({target_x:.1f}, {target_y:.1f}, {target_z:.1f}) mm, "
              f"orient({target_ox:.1f}°, {target_oy:.1f}°, {target_oz:.1f}°)")
        self.error_integral_ = np.zeros(6)
        self.stall_counter = 0

        start_time = time.time()
        iteration = 0
        min_error = float('inf')
        patience = 0
        max_patience = 100

        while time.time() - start_time < timeout:
            iteration += 1
            if self.stepToward(target_ox, target_oy, target_oz, target_x, target_y, target_z):
                self.computeKinematicsAndJacobian()
                error = np.linalg.norm(np.array([target_x, target_y, target_z]) - self.current_position_)
                print(f"Target reached! Error: {error:.2f} mm (iter: {iteration})")
                print(f"Final position: ({self.current_position_[0]:.1f}, {self.current_position_[1]:.1f}, {self.current_position_[2]:.1f}) mm")
                self._stop_all_motors()
                return True

            if iteration % 25 == 0:
                self.computeKinematicsAndJacobian()
                error = np.linalg.norm(np.array([target_x, target_y, target_z]) - self.current_position_)
                print(f"Iter {iteration}: error {error:.1f}mm, pos ({self.current_position_[0]:.1f}, {self.current_position_[1]:.1f}, {self.current_position_[2]:.1f})")

                if error < min_error - 0.1:
                    min_error = error
                    patience = 0
                else:
                    patience += 1
                    if patience > max_patience:
                        print(f"Early stopping - no improvement for {max_patience} checks")
                        break

            time.sleep(self.dt_)

        # 終止時計算殘差
        self.computeKinematicsAndJacobian()
        pos_error = np.linalg.norm(np.array([target_x, target_y, target_z]) - self.current_position_)

        q_target = p.getQuaternionFromEuler([math.radians(target_ox),
                                             math.radians(target_oy),
                                             math.radians(target_oz)])
        q_current = p.getQuaternionFromEuler([float(self.current_orientation_[0]),
                                              float(self.current_orientation_[1]),
                                              float(self.current_orientation_[2])])

        # 四元數誤差 -> 角度
        qe = quat_mul(q_target, quat_conjugate(q_current))
        qw = max(min(qe[3], 1.0), -1.0)
        ang_err = 2.0 * math.acos(qw)
        ang_error_deg = math.degrees(ang_err)

        if pos_error < 5.0 and ang_error_deg < 5.0:
            print(f"Target reached. Position error: {pos_error:.2f} mm, Orientation error: {ang_error_deg:.1f}°")
        else:
            print(f"Timeout. Position error: {pos_error:.2f} mm, Orientation error: {ang_error_deg:.1f}° (iter: {iteration})")

        self._stop_all_motors()
        return pos_error < 5.0 and ang_error_deg < 5.0

    def homePosition(self):
        return self.moveToPosition(0, 0, 200, 0, 0, 0)

    def demo(self):
        print("\n=== Mini Arm Improved Control Demo ===")
        demo_positions = [
            (100, 0, 150, 0, 0, 0),
            (70, 70, 120, 0, 0, 45),
            (0, 100, 150, 0, 0, 90),
            (-70, 70, 180, 0, 0, 135),
            (50, -50, 100, 0, 0, -45),
        ]
        for i, (x, y, z, ox, oy, oz) in enumerate(demo_positions):
            print(f"\n--- Point {i+1}/{len(demo_positions)} ---")
            success = self.moveToPosition(x, y, z, ox, oy, oz, timeout=10.0)
            if success:
                time.sleep(1)
            else:
                print("Continuing to next point...")

        print("\n--- Returning home ---")
        self.homePosition()
        print("Demo complete")

    def interactiveControl(self):
        print("\n=== Mini Arm Improved Velocity Control ===")
        print("Commands:")
        print("  pos x y z [ox oy oz] - Move to position (mm, degrees)")
        print("  demo                 - Run demo sequence")
        print("  home                 - Go home")
        print("  status               - Show current state")
        print("  q                    - Quit")

        while True:
            try:
                self.computeKinematicsAndJacobian()
                pos = self.current_position_
                ori_deg = np.rad2deg(self.current_orientation_)
                print(f"\nPos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) mm, "
                      f"Orient: ({ori_deg[0]:.1f}°, {ori_deg[1]:.1f}°, {ori_deg[2]:.1f}°)")

                cmd = input("Command: ").strip().split()
                if not cmd:
                    continue

                if cmd[0] == 'q':
                    break
                elif cmd[0] == 'pos':
                    if len(cmd) >= 4:
                        x, y, z = map(float, cmd[1:4])
                        ox = oy = oz = 0
                        if len(cmd) >= 7:
                            ox, oy, oz = map(float, cmd[4:7])
                        self.moveToPosition(x, y, z, ox, oy, oz)
                    else:
                        print("Usage: pos x y z [ox oy oz]")
                elif cmd[0] == 'demo':
                    self.demo()
                elif cmd[0] == 'home':
                    self.homePosition()
                elif cmd[0] == 'status':
                    joint_states = [np.rad2deg(self.q_[i]) for i in range(6)]
                    print(f"Joint angles (deg): {[f'{j:.1f}' for j in joint_states]}")
                    print(f"Position (mm): ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
                    print(f"Orientation (deg): ({ori_deg[0]:.1f}, {ori_deg[1]:.1f}, {ori_deg[2]:.1f})")
                else:
                    print("Invalid command")

            except ValueError:
                print("Invalid input")
            except KeyboardInterrupt:
                break

        print("Exiting")

    def __del__(self):
        self.running = False
        if hasattr(self, 'joint_indices'):
            for joint_idx in self.joint_indices[:6]:
                try:
                    p.setJointMotorControl2(self.robotId, joint_idx, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
                except:
                    pass
        if hasattr(self, 'physicsClient'):
            try:
                p.disconnect(self.physicsClient)
            except:
                pass
