# Robot Hand Data Collection and Filtering

This repository contains code for capturing hand pose data using a robot hand model, applying filtering techniques such as Kalman filter and particle filter, and saving the filtered data along with unfiltered angles to a CSV file. The code also includes real-time visualization of hand poses and calculated angles.

## Table of Contents

- [Requirements](#requirements)
- [Usage](#usage)
- [Configuration](#configuration)
- [Results](#results)
- [Data Analysis](#data)
- [Video](#video)

## Requirements

To run the code, you need to have the following dependencies installed:

- pybullet
- opencv-python
- mediapipe
- mpl_toolkits
- numpy
- matplotlib
- pykalman

You can install these dependencies using pip:

## Usage

1. Open the robot_hand.zip file
3. Run Final_Version_Robot_Hand_Simulation.py on your computer to see the unfiltered robot hand simulation
4. The program will start capturing video frames from your webcam and estimate hand poses using MediaPipe. It will then apply Kalman filter and particle filter to filter the hand pose data. The filtered angles, unfiltered angles, and iteration numbers will be saved to a CSV file named `data.csv`.
5. Press 'q' to exit the program.
6. Run KF.py to to see the Kalman filtered robot hand simulation/Run PF.py to to see the Particle filtered robot hand simulation

## Configuration

- You can modify the simulation settings, such as the robot hand model, gravity, and simulation parameters, by changing the code in `main.py`.

- The program uses the `MPL.xml` model file for the robot hand, which is provided by the PyBullet Hand Example repository (https://github.com/philtabor/PyBullet-Hand-Example.git). If you want to use a different model, make sure to update the code accordingly.

- The CSV file where the data is saved can be changed by modifying the `csv_file_data` variable in `main.py`.

## Results

The program calculates the Mean Squared Error (MSE) for both the Kalman filter (KF) and particle filter (PF) to evaluate the filtering performance. The MSE values are saved to separate CSV files named `kf_mse_data.csv` and `pf_mse_data.csv`.

The filtered angles, unfiltered angles, and iteration numbers are saved to a CSV file named `data.csv`. This file contains the following columns:

- Iteration: The iteration number of the data point.
- Index: The index of the joint.
- PF Angle (rad): The filtered angle for the joint using the particle filter (in radians).
- KF Angle (rad): The filtered angle for the joint using the Kalman filter (in radians).
- Unfiltered Angle (rad): The unfiltered angle for the joint (in radians).

You can analyze and visualize the saved data using your preferred data analysis tools.

## Data Analysis

The `data_analysis.ipynb` file contains code for loading the saved data from `data.csv` and visualizing the hand poses, filtered angles, and unfiltered angles using plots and 3D visualizations.

To run the data analysis notebook, make sure you have Jupyter Notebook installed, and then run the following command

## Video

Video Link: https://youtu.be/2MV2wlbh46w
