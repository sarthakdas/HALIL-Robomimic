import json
import os
import subprocess

# Define global variables
algorithm = "kat"
dataset = "lift"

# Function to update JSON file with the algorithm name
def update_json_file(file_path, algorithm_name, dataset_file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        data['algo']["demonstation_data_path"] = dataset_file_path
        
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
        print(f"Updated {file_path} with dataset path {dataset_file_path}")
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the file {file_path}.")

# Function to run terminal command with algorithm and dataset as inputs
def run_terminal_command(algorithm_name, dataset_name):
    command = f"python train.py --config {algorithm_name} --dataset {dataset_name}"
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Command output:\n{result.stdout.decode()}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running command:\n{e.stderr.decode()}")

# Main function
def main():
    json_file_path = "robomimic/exps/templates/XXX.json"
    data_file_path = "tmp/XXX.hdf5"

    # replace XXX with the algorithm name
    json_file_path = json_file_path.replace("XXX", algorithm)
    data_file_path = data_file_path.replace("XXX", dataset)


    
    update_json_file(json_file_path, algorithm, data_file_path)
    run_terminal_command(json_file_path, data_file_path)

if __name__ == "__main__":
    main()
