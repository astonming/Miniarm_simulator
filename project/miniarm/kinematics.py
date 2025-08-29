import numpy as np
import pybullet as p

def solveDLS(J, v, lam_base=0.05):
    """改進的自適應阻尼最小二乘求解器（與原始邏輯一致）"""
    JJT = J @ J.T
    try:
        eigenvalues = np.linalg.eigvalsh(JJT)
        non_zero = eigenvalues[np.abs(eigenvalues) > 1e-12]
        if non_zero.size == 0:
            min_eigen = 1e-12
            max_eigen = 1e-12
        else:
            min_eigen = np.min(np.abs(non_zero))
            max_eigen = np.max(np.abs(eigenvalues))
        condition_number = max_eigen / (min_eigen + 1e-10)

        if condition_number > 1000:
            lam = 0.3
        elif condition_number > 100:
            lam = 0.15
        elif condition_number > 50:
            lam = 0.08
        else:
            lam = lam_base

        if condition_number > 100:
            v = v * 0.5
    except:
        lam = 0.2

    A = JJT + lam**2 * np.eye(6, dtype=np.float32)
    try:
        return J.T @ np.linalg.solve(A, v)
    except:
        try:
            return np.linalg.pinv(J, rcond=0.01) @ v
        except:
            print("Warning: Jacobian solution failed, stopping")
            return np.zeros(6)

def computeKinematicsAndJacobian(robotId, end_effector_index, joint_indices, q_):
    """使用 PyBullet 取得 EE 位置/姿態（mm, rad）與 6x6 Jacobian（角在上、線在下，線速度是 mm/s）"""
    link_state = p.getLinkState(robotId, end_effector_index, computeLinkVelocity=0, computeForwardKinematics=1)
    if link_state is None:
        return None, None, None

    ee_pos_m = np.array(link_state[4])
    ee_orn_quat = link_state[5]
    current_position = (ee_pos_m * 1000.0).astype(np.float32)  # mm
    current_orientation = np.array(p.getEulerFromQuaternion(ee_orn_quat), dtype=np.float32)

    joint_positions = [float(q_[i]) for i in range(len(joint_indices[:6]))]
    zero = [0.0] * len(joint_positions)

    try:
        jac_t, jac_r = p.calculateJacobian(robotId, end_effector_index, [0, 0, 0],
                                           joint_positions, zero, zero)
    except TypeError:
        jac_t, jac_r = p.calculateJacobian(robotId, end_effector_index, [0, 0, 0],
                                           joint_positions, zero, zero)

    Jv = np.array(jac_t, dtype=np.float32) * 1000.0  # m/rad -> mm/rad
    Jw = np.array(jac_r, dtype=np.float32)

    ncols = Jv.shape[1] if Jv.ndim > 1 else 0
    if ncols < 6:
        pad = 6 - ncols
        Jv = np.hstack((Jv, np.zeros((3, pad), dtype=np.float32)))
        Jw = np.hstack((Jw, np.zeros((3, pad), dtype=np.float32)))

    J = np.vstack((Jw, Jv)).astype(np.float32)  # 6x6
    return current_position, current_orientation, J
