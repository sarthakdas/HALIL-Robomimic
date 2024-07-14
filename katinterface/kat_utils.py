import numpy as np
import math

def calculate_triangle_vertices(base, height):
    # Assume isosceles triangle with the base along the x-axis and height along the y-axis
    A = np.array([-base / 2, 0, 0])
    B = np.array([base / 2, 0, 0])
    C = np.array([0, height, 0])
    return np.array([A, B, C])

def procrustes_analysis(A, B):
    # A and B are sets of points
    n = A.shape[0]
    
    # Translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # Singular Value Decomposition
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B.T - R @ centroid_A.T

    return R, t

def transform_vertices(R, t, vertices):
    return (R @ vertices.T).T + t

def optimize_triangle(points, triangle_dimensions):
    base, height = triangle_dimensions
    triangle_vertices = calculate_triangle_vertices(base, height)
    
    # Run Procrustes analysis to find the best alignment
    R, t = procrustes_analysis(triangle_vertices, points)

    # Transform the vertices using the found rotation and translation
    optimized_vertices = transform_vertices(R, t, triangle_vertices)

    return optimized_vertices

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x**2 + z**2), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x**2 + y**2)]
    ])
    return R

def generate_waypoints_quaternion(waypoint):
    x, y, z, qx, qy, qz, qw, gripper = waypoint

    # Quaternion to rotation matrix
    R = quaternion_to_rotation_matrix((qw, qx, qy, qz))

    # Directions
    left_direction = np.array([-0.05, 0, 0])   # Left along negative X
    right_direction = np.array([0.05, 0, 0])   # Right along positive X
    front_direction = np.array([0, 0.1, 0])    # Forward along positive Y

    # Apply rotation
    left_transformed = R @ left_direction
    right_transformed = R @ right_direction
    front_transformed = R @ front_direction

    # Calculate waypoints
    left_waypoint = (x + left_transformed[0], y + left_transformed[1], z + left_transformed[2])
    right_waypoint = (x + right_transformed[0], y + right_transformed[1], z + right_transformed[2])
    front_waypoint = (x + front_transformed[0], y + front_transformed[1], z + front_transformed[2])

    # convert to list for easier access
    left_waypoint = list(left_waypoint)
    right_waypoint = list(right_waypoint)
    front_waypoint = list(front_waypoint)

    return left_waypoint, right_waypoint, front_waypoint, gripper

def generate_waypoints(waypoint):
    x, y, z, roll, pitch, yaw, gripper = waypoint

    # Rotation matrices
    R = np.array([
      [
          math.cos(yaw) * math.cos(pitch),
          math.cos(yaw) * math.sin(pitch) * math.sin(roll) - math.sin(yaw) * math.cos(roll),
          math.cos(yaw) * math.sin(pitch) * math.cos(roll) + math.sin(yaw) * math.sin(roll)
      ],
      [
          math.sin(yaw) * math.cos(pitch),
          math.sin(yaw) * math.sin(pitch) * math.sin(roll) + math.cos(yaw) * math.cos(roll),
          math.sin(yaw) * math.sin(pitch) * math.cos(roll) - math.cos(yaw) * math.sin(roll)
      ],
      [
          -math.sin(pitch),
          math.cos(pitch) * math.sin(roll),
          math.cos(pitch) * math.cos(roll)
      ]
    ])

    # Directions
    left_direction = np.array([-0.05, 0, 0])   # Left along negative X
    right_direction = np.array([0.05, 0, 0])   # Right along positive X
    front_direction = np.array([0, 0.1, 0])    # Forward along positive Y

    # Apply rotation
    left_transformed = R @ left_direction
    right_transformed = R @ right_direction
    front_transformed = R @ front_direction

    # Calculate waypoints
    left_waypoint = (x + left_transformed[0], y + left_transformed[1], z + left_transformed[2])
    right_waypoint = (x + right_transformed[0], y + right_transformed[1], z + right_transformed[2])
    front_waypoint = (x + front_transformed[0], y + front_transformed[1], z + front_transformed[2])

    # convert to list for easier access
    left_waypoint = list(left_waypoint)
    right_waypoint = list(right_waypoint)
    front_waypoint = list(front_waypoint)

    return left_waypoint, right_waypoint, front_waypoint, gripper

def rotation_matrix_from_euler(roll, pitch, yaw):
    R_x = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])
    
    R_y = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    
    R_z = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def euler_from_rotation_matrix(R):
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0
    return roll, pitch, yaw

def inverse_generate_waypoints(front_waypoint, right_waypoint, left_waypoint):
    # Calculate the central point
    x = (left_waypoint[0] + right_waypoint[0]) / 2
    y = (left_waypoint[1] + right_waypoint[1]) / 2
    z = (left_waypoint[2] + right_waypoint[2]) / 2

    # Calculate the transformed vectors
    left_transformed = np.array(left_waypoint) - np.array([x, y, z])
    right_transformed = np.array(right_waypoint) - np.array([x, y, z])
    front_transformed = np.array(front_waypoint) - np.array([x, y, z])

    # Directions (normalized)
    left_direction = left_transformed / np.linalg.norm(left_transformed)
    right_direction = right_transformed / np.linalg.norm(right_transformed)
    front_direction = front_transformed / np.linalg.norm(front_transformed)

    # Construct the rotation matrix from direction vectors
    R_inv = np.vstack([right_direction, front_direction, np.cross(right_direction, front_direction)]).T

    # Extract roll, pitch, yaw from the rotation matrix
    roll, pitch, yaw = euler_from_rotation_matrix(R_inv)

    return x, y, z, roll, pitch, yaw

def generate_original_waypoints(points):
    '''
    Generate the original waypoints from the triangle vertices

    Parameters:

    points (np.array): The triangle vertices only takes in x,y,z remove the gripper value before passing it to this function

    Returns:

    tuple: The original waypoints
    '''
    triangle_vertices = (0.1, 0.1)  # Example triangle dimensions (base, height)
    adjusted_vertices = optimize_triangle(points, triangle_vertices).round(2)
    new_waypoint = inverse_generate_waypoints(adjusted_vertices[0], adjusted_vertices[1], adjusted_vertices[2])
    # convert to list 
    new_waypoint = list(new_waypoint)
    return new_waypoint

if __name__ == '__main__':
    # Example usage
    waypoint = [0, 0, 0, 0, 0, np.pi /4, 0] 
    print('Original waypoint:', waypoint)
    left_waypoint, right_waypoint, front_waypoint, gripper = generate_waypoints(waypoint)
    print('Left waypoint:', [round(coord, 3) for coord in front_waypoint])
    print('Right waypoint:', right_waypoint)
    print('Front waypoint:', front_waypoint)
    new_waypoint = inverse_generate_waypoints(left_waypoint, right_waypoint, front_waypoint)
    print('New waypoint:', new_waypoint)

