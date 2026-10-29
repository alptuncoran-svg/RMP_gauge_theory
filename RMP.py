import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

# ==========================================
# 1. Dual Quaternion & Lie Group Math
# ==========================================

def q_mul(q1, q2):
    s1, u1, v1, w1 = q1
    s2, u2, v2, w2 = q2
    
    s3 = s1 * s2 - u1 * u2 - v1 * v2 - w1 * w2
    u3 = s1 * u2 + s2 * u1 + v1 * w2 - w1 * v2
    v3 = s1 * v2 + s2 * v1 + w1 * u2 - u1 * w2
    w3 = s1 * w2 + s2 * w1 + u1 * v2 - v1 * u2
    
    return np.array([s3, u3, v3, w3]) 

def rot(u, theta, q):
    st, ct = np.sin(theta/2), np.cos(theta/2)
    qr = np.array([ct, u[0]*st, u[1]*st, u[2]*st])
    qrinv = np.array([ct, -u[0]*st, -u[1]*st, -u[2]*st])
    return q_mul(q_mul(qr, q), qrinv)

def q_inv(q):
    s, u, v, w = q
    return np.array([s, -u, -v, -w])

def motor(r, u, t, d):
    s = np.sin(t/2)
    c = np.cos(t/2)
    m = np.cross(r, u)
    dh = d/2
    
    qr = np.array([c, s*u[0], s*u[1], s*u[2]])
    qd = np.array([-s*dh, 
                   u[0]*c*dh + m[0]*s, 
                   u[1]*c*dh + m[1]*s, 
                   u[2]*c*dh + m[2]*s])
    return [qr, qd]

def dq_mul(dq1, dq2):
    q1r, q1d = dq1
    q2r, q2d = dq2
    
    q3r = q_mul(q1r, q2r)
    q3d = q_mul(q1r, q2d) + q_mul(q1d, q2r) 
    
    return [q3r, q3d]


# ==========================================
# 2. Robot Kinematics (Forward)
# ==========================================

def forward(thetas, joint_axes, joint_positions):
    M_ee = [np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 0.0])]
    for i in range(len(thetas)):
        M_joint = motor(joint_positions[i], joint_axes[i], thetas[i], d=0.0)
        M_ee = dq_mul(M_ee, M_joint)
    return M_ee

def transform_point(M, point):
    pt_dq = [np.array([1.0, 0.0, 0.0, 0.0]), 
             np.array([0.0, point[0], point[1], point[2]])]
    # Conjugate for point transformation: Q_r* - epsilon Q_d*
    M_conj = [q_inv(M[0]), -q_inv(M[1])]
    transformed_pt_dq = dq_mul(dq_mul(M, pt_dq), M_conj)
    return transformed_pt_dq[1][1:4]

def get_joints(thetas, joint_axes, joint_positions, initial_ee_pos):
    positions = [np.array([0.0, 0.0, 0.0])] 
    M_accum = [np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 0.0])]
    
    for i in range(len(thetas)):
        if i > 0:
            pos = transform_point(M_accum, joint_positions[i])
            positions.append(pos)
        M_joint = motor(joint_positions[i], joint_axes[i], thetas[i], d=0.0)
        M_accum = dq_mul(M_accum, M_joint)
        
    ee_pos = transform_point(M_accum, initial_ee_pos)
    positions.append(ee_pos)
    return np.array(positions)


# ==========================================
# 3. O(n) Geometric Jacobians (J & Jdot)
# ==========================================

def spatial_jacobian(thetas, theta_dots, joint_axes, joint_positions):
    n = len(thetas)
    J = np.zeros((6, n))
    Jdot = np.zeros((6, n))
    
    M_accum = [np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 0.0])]
    w_accum = np.zeros(3)
    v_accum = np.zeros(3)
    
    for i in range(n):
        u = joint_axes[i]
        r = joint_positions[i]
        
        v = np.cross(r, u)
        S_dq = [np.array([0.0, u[0], u[1], u[2]]), np.array([0.0, v[0], v[1], v[2]])]
        
        M_conj = [q_inv(M_accum[0]), q_inv(M_accum[1])]
        S_current = dq_mul(dq_mul(M_accum, S_dq), M_conj)
        
        w_current = S_current[0][1:4]
        v_current = S_current[1][1:4]
        
        J[0:3, i] = w_current
        J[3:6, i] = v_current
        
        w_dot = np.cross(w_accum, w_current)
        v_dot = np.cross(w_accum, v_current) + np.cross(v_accum, w_current)
        
        Jdot[0:3, i] = w_dot
        Jdot[3:6, i] = v_dot
        
        w_accum = w_accum + w_current * theta_dots[i]
        v_accum = v_accum + v_current * theta_dots[i]
        
        M_joint = motor(r, u, thetas[i], 0.0)
        M_accum = dq_mul(M_accum, M_joint)
        
    return J, Jdot

def ee_jacobian(J_spatial, Jdot_spatial, ee_pos, theta_dots):
    n = J_spatial.shape[1]
    J_ee = np.zeros((6, n))
    Jdot_ee = np.zeros((6, n))
    
    w_ee_global = J_spatial[0:3, :] @ theta_dots
    v_s_global = J_spatial[3:6, :] @ theta_dots
    p_dot = v_s_global + np.cross(w_ee_global, ee_pos)
    
    for i in range(n):
        w = J_spatial[0:3, i]
        v_s = J_spatial[3:6, i]
        w_dot = Jdot_spatial[0:3, i]
        v_s_dot = Jdot_spatial[3:6, i]
        
        J_ee[0:3, i] = w
        J_ee[3:6, i] = v_s + np.cross(w, ee_pos)
        
        Jdot_ee[0:3, i] = w_dot
        Jdot_ee[3:6, i] = v_s_dot + np.cross(w_dot, ee_pos) + np.cross(w, p_dot)
        
    return J_ee, Jdot_ee


# ==========================================
# 4. True Second-Order RMP Controller
# ==========================================

def rmp_accel(J, Jdot, theta_dots, error, alpha=0.01, beta=1.0, gain=10.0, damping=5.0):
    """
    Calculates q_ddot using the full M(q)q_ddot + C(q, q_dot)q_dot = tau dynamics.
    """
    error_norm = np.linalg.norm(error)
    
    if error_norm < 1e-6:
        return np.zeros(J.shape[1])
        
    # Task space desired acceleration 
    a = gain * (1.0 - np.exp(-beta * (error_norm ** 2))) * (error / error_norm)
    
    manipulability = J @ J.T
    identity_task = np.eye(J.shape[0])
    
    # Gain co-tensor K
    K_inv = manipulability + (identity_task * alpha)
    K = np.linalg.inv(K_inv)
    
    # Derivative of K
    K_inv_dot = (Jdot @ J.T) + (J @ Jdot.T)
    K_dot = -K @ K_inv_dot @ K
    
    # Vielbein embedding L and its derivative
    L = K @ J
    L_dot = (K_dot @ J) + (K @ Jdot)
    
    # Metric Mass and Coriolis 
    M = L.T @ L
    C = L.T @ L_dot
    
    # Pullback torque
    tau_pullback = L.T @ a
    
    # Physical damping applied as a force
    damping_force = damping * theta_dots
    
    # Solve for acceleration: M * q_ddot + C * q_dot = tau - damping
    lambda_inertia = 0.1 # Base inertia for null-space stability
    identity_joint = np.eye(J.shape[1])
    M_invertible = M + (identity_joint * lambda_inertia)
    
    net_torque = tau_pullback - (C @ theta_dots) - damping_force
    q_ddot = np.linalg.inv(M_invertible) @ net_torque
    
    return q_ddot


# ==========================================
# 5. Simulation Main Loop & Plotting
# ==========================================

if __name__ == "__main__":
    i_axis = np.array([1.0, 0.0, 0.0])
    k_axis = np.array([0.0, 0.0, 1.0])
    robot_joint_axes = [k_axis, i_axis, k_axis, i_axis, k_axis, i_axis, k_axis]
    
    link_lengths = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.1]
    robot_joint_positions = []
    current_z = 0.0
    for length in link_lengths:
        robot_joint_positions.append(np.array([0.0, 0.0, current_z]))
        current_z += length 
        
    initial_ee_pos = np.array([0.0, 0.0, current_z])

    thetas = np.array([0.0, 0.785, 0.0, 1.57, 0.0, 0.785, 0.0])
    theta_dots = np.zeros(7)
    
    target_pos = np.array([0.4, 0.4, 0.6])
    
    # RMP Parameters tuned for FAST Second-Order convergence
    alpha = 0.005  
    beta = 10
    gain =   30
    damping = 3*np.sqrt(gain)
    dt = 0.01 
    
    print(f"Target Position: {target_pos}")
    
    ee_trajectory = []
    history_arm_points = []
    
    for step in range(500):
        final_motor = forward(thetas, robot_joint_axes, robot_joint_positions)
        current_ee_pos = transform_point(final_motor, initial_ee_pos)
        
        arm_pts = get_joints(thetas, robot_joint_axes, robot_joint_positions, initial_ee_pos)
        history_arm_points.append(arm_pts)
        ee_trajectory.append(current_ee_pos)
        
        pos_error = target_pos - current_ee_pos
        error_magnitude = np.linalg.norm(pos_error)
        
        if error_magnitude < 0.01:
            print(f"Converged at step {step}! Final Pos: {current_ee_pos}")
            break
            
        J_spatial, Jdot_spatial = spatial_jacobian(
            thetas, theta_dots, robot_joint_axes, robot_joint_positions
        )
        J_ee, Jdot_ee = ee_jacobian(
            J_spatial, Jdot_spatial, current_ee_pos, theta_dots
        )
        
        J_pos = J_ee[3:6, :]
        Jdot_pos = Jdot_ee[3:6, :]
        
        # Pullback acceleration directly via the Vielbein Mass & Coriolis matrices
        q_ddot = rmp_accel(
            J_pos, Jdot_pos, theta_dots, pos_error, alpha=alpha, beta=beta, gain=gain, damping=damping
        )
        
        # Second-order physics integration
        theta_dots += q_ddot * dt
        thetas += theta_dots * dt
        
        if step % 20 == 0:
            print(f"Step {step:03d} | Error: {error_magnitude:.4f} | EE Pos: {current_ee_pos}")

    # ==========================================
    # 6. Interactive Visualization
    # ==========================================
    ee_trajectory = np.array(ee_trajectory)
    history_arm_points = np.array(history_arm_points)
    num_steps = len(history_arm_points)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.25) 
    
    ax.plot(ee_trajectory[:, 0], ee_trajectory[:, 1], ee_trajectory[:, 2], 
            label='EE Trajectory', color='blue', linewidth=2, linestyle='--', alpha=0.5)
    ax.scatter(*target_pos, color='red', s=150, label='Target', marker='*')
    
    initial_pts = history_arm_points[-1]
    arm_line, = ax.plot(initial_pts[:, 0], initial_pts[:, 1], initial_pts[:, 2], 
                        label='Arm Pose', color='black', marker='o', linewidth=3, markersize=6)
    ax.scatter(0, 0, 0, color='green', s=100, label='Base', marker='s')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('RMP Arm Kinematics (Use Slider to Replay)')
    ax.legend()
    
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    step_slider = Slider(
        ax=ax_slider,
        label='Simulation Step',
        valmin=0,
        valmax=num_steps - 1,
        valinit=num_steps - 1,
        valstep=1
    )

    def update_plot(val):
        idx = int(step_slider.val)
        pts = history_arm_points[idx]
        arm_line.set_data(pts[:, 0], pts[:, 1])
        arm_line.set_3d_properties(pts[:, 2])
        fig.canvas.draw_idle()

    step_slider.on_changed(update_plot)
    plt.show()