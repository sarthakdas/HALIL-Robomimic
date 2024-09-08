import h5py
import json
import shutil

def read_env_args(file_path):
    with h5py.File(file_path, 'r') as file:
        env_args_json = file['data'].attrs['env_args']
        env_args = json.loads(env_args_json)
    return env_args

def write_env_args(file_path, new_env_args):
    with h5py.File(file_path, 'a') as file:
        new_env_args_json = json.dumps(new_env_args, indent=4)
        file['data'].attrs['env_args'] = new_env_args_json

def save_updated_file(original_file_path, updated_file_path):
    shutil.copyfile(original_file_path, updated_file_path)
    return updated_file_path

def main():
    original_file_path = 'tmp/lift.hdf5'
    updated_file_path = 'tmp/lift_ik.hdf5'

    # Save the original file to a new file
    updated_file_path = save_updated_file(original_file_path, updated_file_path)

    # Read current env_args
    env_args = read_env_args(updated_file_path)
    print("Current env_args:")
    print(json.dumps(env_args, indent=4))
    
    # Modify env_args as needed
    # Example: change controller type
    env_args['env_kwargs']['controller_configs']['type'] = 'IK_POSE'
    env_args['env_kwargs']['controller_configs']['ik_pos_limit'] = 0.02
    env_args['env_kwargs']['controller_configs']['ik_ori_limit'] = 0.05

    # delete unnecessary keys
    del env_args['env_kwargs']['controller_configs']['input_min']
    del env_args['env_kwargs']['controller_configs']['input_max']
    del env_args['env_kwargs']['controller_configs']['output_min']
    del env_args['env_kwargs']['controller_configs']['output_max']
    del env_args['env_kwargs']['controller_configs']['impedance_mode']
    del env_args['env_kwargs']['controller_configs']['control_delta']
    del env_args['env_kwargs']['controller_configs']['uncouple_pos_ori']
    
    # Write the modified env_args back to the new file
    write_env_args(updated_file_path, env_args)
    print(f"Updated env_args written to {updated_file_path}.")

    print("Updated env_args:")
    print(json.dumps(env_args, indent=4))

if __name__ == "__main__":
    main()
