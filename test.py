import h5py


def get_trajectory(file_path, demo_number):
    trajectory = {}
    demo_key = f"data/demo_{demo_number}"

    with h5py.File(file_path, 'r') as file:
        if demo_key not in file:
            raise ValueError(f"Demo {demo_number} not found in the file.")

        demo_group = file[demo_key]

        trajectory['actions'] = demo_group['actions'][:]
        trajectory['dones'] = demo_group['dones'][:]
        trajectory['rewards'] = demo_group['rewards'][:]
        trajectory['states'] = demo_group['states'][:]

        trajectory['obs'] = {
            name: demo_group['obs'][name][:]
            for name in demo_group['obs']
        }
        trajectory['next_obs'] = {
            name: demo_group['next_obs'][name][:]
            for name in demo_group['next_obs']
        }

    return trajectory


# Example usage
file_path = 'tmp/test_v141.hdf5'
demo_number = 5
trajectory = get_trajectory(file_path, demo_number)

# round trajaectory actions to 2 decimal places
trajectory['actions'] = trajectory['actions'].round(3)


# Display the trajectory
import pprint

print(len(trajectory['actions']))
# save the trajectory to a file
with open('_trajectory.txt', 'w') as f:
    pprint.pprint(trajectory, f)
# [ 1.54274528e-02, -2.29044097e-02,  8.31156608e-01,
        #  0.00000000e+00,  0.00000000e+00,  9.00708631e-01,
        # -4.34423713e-01, -1.04471783e-01,  5.48668298e-03,
        #  1.82095699e-01]