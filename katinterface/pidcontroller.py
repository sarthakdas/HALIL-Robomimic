import math
import numpy as np
import quaternion

class PController6D:
    def __init__(self, Kp_pos, Kp_ori, epsilon=1e-6, desired_x=0.0, desired_y=0.0, desired_z=0.0, desired_roll=0.0, desired_pitch=0.0, desired_yaw=0.0):
        self.Kp_pos = Kp_pos  # Proportional gain for position (x, y, z)
        self.Kp_ori = Kp_ori  # Proportional gain for orientation (ax, ay, az)
        self.epsilon = epsilon  # Threshold to ignore small errors
        self.desired_pos = np.array([desired_x, desired_y, desired_z])  # Desired position
        self.desired_orn = self.rpy_to_quaternion(desired_roll, desired_pitch, desired_yaw)  # Desired orientation
    
    def rpy_to_quaternion(self, roll, pitch, yaw):
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        q = np.quaternion(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        )
        return q
    
    def quaternion_error(self, q1, q2):
        return q2 * q1.conjugate()

    def quaternion_to_axis_angle(self, q):
        # Clamp q.w to the range [-1, 1] to avoid invalid values in arccos
        q_w_clamped = np.clip(q.w, -1.0, 1.0)
        angle = 2 * np.arccos(q_w_clamped)
        s = np.sqrt(1 - q_w_clamped * q_w_clamped)
        if s < 1e-6:
            x, y, z = q.x, q.y, q.z
        else:
            x, y, z = q.x / s, q.y / s, q.z / s
        return np.array([x, y, z]) * angle

    def set_tuning(self, Kp_pos, Kp_ori):
        self.Kp_pos = Kp_pos
        self.Kp_ori = Kp_ori
    
    def set_target(self, target_pos_orn):
        self.desired_pos = target_pos_orn[:3]
        self.desired_orn = self.rpy_to_quaternion(*target_pos_orn[3:])

    def _wrap_angle(self, angle):
        return ((angle + math.pi) % (2 * math.pi)) - math.pi
    
    def __call__(self, current_pos_orn):
        """
        Calculates the control outputs (motor powers) based on the current state and desired setpoints.
        
        Args:
            current_pos_orn: np.array: Current position and orientation of the robot [x, y, z, qx, qy, qz, qw].
        
        Returns:
            list: Control outputs [motor_power_x, motor_power_y, motor_power_z, motor_power_ax, motor_power_ay, motor_power_az].
        """
        current_pos_orn = np.array(current_pos_orn)
        current_pos = current_pos_orn[:3]
        current_orn = np.quaternion(current_pos_orn[6], current_pos_orn[3], current_pos_orn[4], current_pos_orn[5])

        # Calculate the error for each position axis
        pos_error = self.desired_pos - current_pos
        orient_error = self.quaternion_error(current_orn, self.desired_orn)

        orient_error_axis_angle = self.quaternion_to_axis_angle(orient_error)

        # Apply threshold to ignore small errors
        orient_error_axis_angle[np.abs(orient_error_axis_angle) < self.epsilon] = 0

        control_action_pos = self.Kp_pos * pos_error
        control_action_orient = self.Kp_ori * orient_error_axis_angle

        motor_power_x, motor_power_y, motor_power_z = control_action_pos
        motor_power_ax, motor_power_ay, motor_power_az = control_action_orient

        return [motor_power_x, motor_power_y, motor_power_z, motor_power_ax, motor_power_ay, motor_power_az]

def quaternion_to_rpy(q):
    # Ensure the quaternion is normalized
    q = q / np.linalg.norm([q.w, q.x, q.y, q.z])
    
    # Extract the Euler angles from the quaternion
    roll = np.arctan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
    pitch = np.arcsin(2 * (q.w * q.y - q.z * q.x))
    yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    
    return roll, pitch, yaw

def convert_to_xyz_rpy(x, y, z, qx, qy, qz, qw):
    q = np.quaternion(qw, qx, qy, qz)
    roll, pitch, yaw = quaternion_to_rpy(q)
    return np.array([x, y, z, roll, pitch, yaw])

if __name__ == "__main__":
    # Example usage:
    Kp_pos = 1.0  # Proportional gain for position (x, y, z)
    Kp_ori = 0.1  # Proportional gain for orientation (ax, ay, az)
    epsilon = 1e-6  # Threshold to ignore small errors
    p_controller_6d = PController6D(Kp_pos, Kp_ori, epsilon)

    # Set the desired target coordinates and orientation
    target_pos_orn = np.array([-0.1, -0.0, 1.01, 0.0, 0.15, 0.0, 1.0])
    

    p_controller_6d.set_target(convert_to_xyz_rpy(*target_pos_orn))

    # Assume current_x, current_y, current_z, current_qx, current_qy, current_qz, current_qw are the current coordinates and orientation of the robot
    current_pos_orn = np.array([-0.1, -0.0, 1.01, 0.0, 0.14, 0.0, 1.0])

    # Call the controller to get motor power for each axis
    motor_power_x, motor_power_y, motor_power_z, motor_power_ax, motor_power_ay, motor_power_az = p_controller_6d(
        current_pos_orn
    )

    print(f"Motor Power X: {motor_power_x}")
    print(f"Motor Power Y: {motor_power_y}")
    print(f"Motor Power Z: {motor_power_z}")
    print(f"Motor Power Ax: {motor_power_ax}")
    print(f"Motor Power Ay: {motor_power_ay}")
    print(f"Motor Power Az: {motor_power_az}")
