# Robomimic Setup Guide

## 1. Installation

To set up Robomimic, follow the installation instructions in the official [Robomimic GitHub repository](https://github.com/ARISE-Initiative/robomimic#installation).

## 2. Prepare Demonstration Data

1. Create a directory named `tmp` in your working directory:
   ```bash
   mkdir tmp
   ```
2. Place your demonstration data inside the `tmp` folder.

## 3. Configure Training

1. Open your `kat.json` configuration file.
2. Ensure the `dataset_path` parameter in the configuration file points to the demonstration data in the `tmp` folder

## 4. Run Training

1. Run the `train.py` script with the modified configuration

Make sure the `config_path` in `train.py` is correctly set to your `kat.json` file.

For detailed instructions, refer to the [Robomimic documentation](https://arise-initiative.github.io/robomimic/).
