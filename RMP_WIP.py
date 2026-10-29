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

def rmp_accel(J, Jdot, theta_dots, error, r, r_obs, alpha=0.01, beta=1.0, gain=10.0, damping=5.0, h=5.0, k_repulse=0.05, delta=0.05, obs_radius=0.2):
    """
    Calculates q_ddot using the full M(q)q_ddot + C(q, q_dot)q_dot = tau dynamics.
    Includes a Gaussian tapering function so the obstacle's influence drops to 0 at a distance.
    """
    n_joints = J.shape[1]
    error_norm = np.linalg.norm(error)
    
    # -------------------------------------
    # 1. Target RMP 
    # -------------------------------------
    if error_norm > 1e-6:
        a_target = gain * (1.0 - np.exp(-beta * (error_norm ** 2))) * (error / error_norm)
        K_inv_target = (J @ J.T) + (np.eye(J.shape[0]) * alpha)
        K_target = np.linalg.inv(K_inv_target)
        K_dot_target = -K_target @ (Jdot @ J.T + J @ Jdot.T) @ K_target
        
        # Correct Riemannian pullback (J.T @ K @ J)
        M_target = J.T @ K_target @ J
        C_target = J.T @ K_dot_target @ J + J.T @ K_target @ Jdot
        tau_target = J.T  @ (K_target @ a_target)
    else:
        M_target = np.zeros((n_joints, n_joints))
        C_target = np.zeros((n_joints, n_joints))
        tau_target = np.zeros(n_joints)

    # -------------------------------------
    # 2. Obstacle RMP (With Brakes & Tapering)
    # -------------------------------------
    x = r - r_obs
    d = np.linalg.norm(x)
    
    M_obs = np.zeros((n_joints, n_joints))
    C_obs = np.zeros((n_joints, n_joints))
    tau_obs = np.zeros(n_joints)
    
    if d > 1e-6:
        v = J @ theta_dots
        n = x / d
        
        d_dot = np.dot(n, v)
        n_dot = (v - n * d_dot) / d
        
        # Smooth Gaussian tapering based on distance
        w = np.exp(-(d**2) / (2 * obs_radius**2))
        w_dot = -(d * d_dot / (obs_radius**2)) * w
        
        # Base Metric and Derivative
        kr_base = h / (d + delta)
        kr_dot_base = -h * d_dot / ((d + delta) ** 2)
        
        # Apply Tapering Weight (Product rule for derivative)
        kr = kr_base * w
        kr_dot = kr_dot_base * w + kr_base * w_dot
        
        N = np.outer(n, n)
        N_dot = np.outer(n_dot, n) + np.outer(n, n_dot)
        
        K_obs = kr * N
        K_dot_obs = (kr_dot * N) + (kr * N_dot)
        
        # Apply the identical tapering to the explicit repulsion force
        a_obs = (k_repulse / (d**2)) * w * n
        
        # Correct Riemannian pullback (J.T @ K @ J)
        M_obs = J.T @ K_obs @ J
        C_obs = J.T @ K_dot_obs @ J + J.T @ K_obs @ Jdot
        tau_obs = J.T @ (K_obs @ a_obs)

    # -------------------------------------
    # 3. Summation & Physics Solve
    # -------------------------------------
    lambda_inertia = 0.1 
    M_total = M_target + M_obs + (np.eye(n_joints) * lambda_inertia)
    C_total = C_target + C_obs
    tau_total = tau_target + tau_obs
    
    damping_force = damping * theta_dots
    net_torque = tau_total - (C_total @ theta_dots) - damping_force
    
    q_ddot = np.linalg.inv(M_total) @ net_torque
    
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
    obstacle_pos = np.array([0.25, 0.15, 0.8])
    
    # Target Parameters
    alpha = 0.005  
    beta = 10  
    gain = 100    
    damping = 2*np.sqrt(gain)
    
    # Obstacle Parameters
    h_val = 2           # Metric strength (Brakes/Steering)
    k_repulse = 1        # Repulsion strength (Push)
    delta_val = 0.05       # Singularity softening
    obs_radius_val = 0.1  # The "danger zone" radius where tapering begins
    
    dt = 0.01 
    
    print(f"Target Position: {target_pos}")
    print(f"Obstacle Position: {obstacle_pos}")
    
    ee_trajectory = []
    history_arm_points = []
    min_obs_dist = float('inf')
    
    for step in range(500):
        final_motor = forward(thetas, robot_joint_axes, robot_joint_positions)
        current_ee_pos = transform_point(final_motor, initial_ee_pos)
        
        arm_pts = get_joints(thetas, robot_joint_axes, robot_joint_positions, initial_ee_pos)
        history_arm_points.append(arm_pts)
        ee_trajectory.append(current_ee_pos)
        
        dist_to_obs = np.linalg.norm(current_ee_pos - obstacle_pos)
        if dist_to_obs < min_obs_dist:
            min_obs_dist = dist_to_obs
            
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
        
        # Calculate full acceleration via combined RMP logic with local tapering
        q_ddot = rmp_accel(
            J_pos, Jdot_pos, theta_dots, pos_error, 
            current_ee_pos, obstacle_pos, 
            alpha=alpha, beta=beta, gain=gain, damping=damping, 
            h=h_val, k_repulse=k_repulse, delta=delta_val, obs_radius=obs_radius_val
        )
        
        # Second-order physics integration
        theta_dots += q_ddot * dt
        thetas += theta_dots * dt
        
        if step % 20 == 0:
            print(f"Step {step:03d} | Err: {error_magnitude:.4f} | Obs Dist: {dist_to_obs:.4f}")
            
    print("-" * 40)
    print(f"Simulation Complete. Minimum distance to obstacle: {min_obs_dist:.4f} m")
    print("-" * 40)

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
    ax.scatter(*obstacle_pos, color='orange', s=200, label='Obstacle', marker='o')
    
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