

import numpy as np
from scipy.spatial.transform import Rotation as R
import katinterface.kat_utils as KATUtils
import torch

def quaternion_inverse(q):
    # Inverse of a quaternion q = (w, x, y, z) is given by q_inv = (w, -x, -y, -z)
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quaternion_multiply(q1, q2):
    # Quaternion multiplication
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])

def quaternion_to_euler(q):
    # Convert quaternion to Euler angles (roll, pitch, yaw)
    r = R.from_quat(q)
    return r.as_euler('xyz', degrees=True)

def decimal_place_converter(list_of_values, mode="encode"):
    return [round(coord, 3) for coord in list_of_values]
    if mode == "encode":
        # mulitply all values by 100 to eliminate decimal place
        list_of_values = [int(coord * 100) for coord in list_of_values]
        # remove decial place values
        list_of_values = [round(coord, 0) for coord in list_of_values]
        return list_of_values
    elif mode == "decode":
        return [coord / 100 for coord in list_of_values]

def input_formatter(obs, decimal_places):
    # round all the values in the observation dictionary and round them to the specified decimal places

    obs["object"] = decimal_place_converter(obs["object"])
    obs["object_orientation"] = decimal_place_converter(obs["object_orientation"])
    obs["object_to_eef_position"] = decimal_place_converter(obs["object_to_eef_position"])
    obs["object_to_eef_orientation"] = decimal_place_converter(obs["object_to_eef_orientation"])

    del obs["object_to_eef_orientation"]
    del obs["object_to_eef_position"]

    return obs

def output_formatter(demo_data, decimal_places, sample_frequency):
    actions = demo_data["actions"]
    gripper_values = actions[:, -1]

    robot_ee_position = demo_data["obs"]["robot0_eef_pos"][::sample_frequency]
    robot_ee_quaternion = demo_data["obs"]["robot0_eef_quat"][::sample_frequency]

    robot_ee_pos_quate_gripper = np.concatenate((robot_ee_position, robot_ee_quaternion, gripper_values[:, None]), axis=1)
    robot_ee_pos_quate_gripper = np.round(robot_ee_pos_quate_gripper, decimal_places)

    return robot_ee_pos_quate_gripper

def gripper_0_to_1(gripper, mode="encode"):
    if mode == "encode":
        if gripper == -1:
            return 0
        else:
            return 1
    elif mode == "decode":
        if gripper == 0:
            return -1
        else:
            return 1

def action_token_formatter(waypoints_trajectory):
    kat_actions = []
    for robot_state in waypoints_trajectory:

        left_position, right_position, front_position, gripper = KATUtils.generate_waypoints_quaternion(robot_state)

        gripper = gripper_0_to_1(gripper, mode="encode")

        # mulitply all values by 100 to eliminate decimal place 
        front_position = decimal_place_converter(front_position)
        right_position = decimal_place_converter(right_position)
        left_position = decimal_place_converter(left_position)

        # combine as a list
        kat_action = [front_position, right_position, left_position, gripper]
        kat_actions.append(kat_action)

    return kat_actions

def obs_to_obsdict(obs):
    obs_position = obs[0:3]
    obs_orientation = obs[3:7] # quaterion
    obs_obj_to_ee_position = obs[7:10]
    obs_obj_to_ee_orientation = obs[10:14] # quaternion

    return {
        "object": obs_position.tolist(),
        "object_orientation": obs_orientation.tolist(),
        "object_to_eef_position": obs_obj_to_ee_position.tolist(),
        "object_to_eef_orientation": obs_obj_to_ee_orientation.tolist()
    }

def demo_data_to_prompt_format(demo_data,sample_frequency,decimal_places):
    '''
    takes in a single full demonstation dictionary and converts it to demonstartion prompt format
    
    Input: data: dictionary with keys 'states', 'actions', 'quaternions', 'gripper', 'object'
           sample_frequency: int, the frequency at which the data was sampled
           decimal_places: int, the number of decimal places to round to

    Ouput: 
    '''
    demo_data["actions"] = demo_data["actions"][::sample_frequency]

    obs_dict = obs_to_obsdict(demo_data["obs"]["object"][0])
    starting_observations = input_formatter(obs_dict, decimal_places)

    waypoints_trajectory = output_formatter(demo_data, decimal_places, sample_frequency)

    action_token_trajectory = action_token_formatter(waypoints_trajectory)

    return starting_observations, waypoints_trajectory, action_token_trajectory

def action_token_to_waypoint(action_token_trajectory):
    waypoints_trajectory = []
    for action_token in action_token_trajectory:
        front_position = action_token[0]
        right_position = action_token[1]
        left_position = action_token[2]
        gripper = action_token[3]

        front_position = decimal_place_converter(front_position, mode="decode")
        right_position = decimal_place_converter(right_position, mode="decode")
        left_position = decimal_place_converter(left_position, mode="decode")

        gripper = gripper_0_to_1(gripper, mode="decode")

        robot_state = KATUtils.inverse_generate_waypoints(np.array(front_position), np.array(right_position), np.array(left_position))
        robot_state = np.concatenate((robot_state, [gripper]), axis=0)
        waypoints_trajectory.append(robot_state)

    return waypoints_trajectory

def rollout_action_to_input_format(rollout_action, decimal_places):

    rollout_action = rollout_action[0]
    rollout_action = obs_to_obsdict(rollout_action)
    rollout_action = input_formatter(rollout_action, decimal_places)
    return rollout_action