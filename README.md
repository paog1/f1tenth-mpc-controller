# Iterative Racing-Line Learning MPC for F1TENTH

[![F1TENTH Simulation Video](https://img.youtube.com/vi/PZtI_nIbe9g/maxresdefault.jpg)](https://www.youtube.com/watch?v=PZtI_nIbe9g)
*Click the image above to view the simulation progress video on YouTube.*

## Project Overview
This repository contains the implementation of an Iterative Racing-Line Learning Model Predictive Controller (MPC) for the F1TENTH platform. The project is developed in **ROS 2 Foxy** using **Python** and relies on the **OSQP** solver for real-time optimization. 

A core philosophy of this project is the strict adherence to a **purely mathematical and iterative approach**. The learning mechanism relies entirely on global continuous optimization and dynamic curvature calculations, explicitly avoiding the use of Artificial Intelligence (AI) or localized track-sector heuristics. 

Furthermore, the controller is designed with advanced soft constraints (slack variables) that allow the vehicle to safely touch or graze track boundaries without the solver failing (becoming infeasible), promoting aggressive and continuous learning.

## Repository Structure & Core Components

To navigate this repository effectively, please focus on the following four critical Python scripts within the `mpc_controller` package. These files constitute the entirety of the project's logic and iterative learning pipeline.

### 1. `mpc_controller.py` (The Main Node)
This is the primary executable file that drives the vehicle. It integrates the learning managers and the real-time MPC loop.
* **Functionality:** Linearizes the Discrete Kinematic Bicycle Model, assembles the sparse QP matrices (handling `r_delta` and `q_ey` steering weight balancing), and publishes optimal `delta` (steering) and `a` (acceleration) commands at 10Hz.
* **Significance:** This is the only file required to actively run the autonomous navigation in the simulator.

### 2. `MPC.py` (Standalone Controller Validation)
A standalone script isolated from the iterative learning managers. 
* **Functionality:** Implements the base MPC logic strictly for trajectory tracking.
* **Significance:** Used to tune the controller matrices and validate that the underlying mathematical model can follow a static reference line accurately and stably before introducing dynamic path mutations.

### 3. `RacingLineManager.py` (Path Optimization)
The core of the geometric learning engine.
* **Functionality:** Analyzes telemetry from the previously completed lap and generates the next candidate racing line. It mathematically mutates the line toward the apexes and passes it through a global Minimum Curvature QP solver to ensure physical viability.
* **Significance:** Proves the iterative nature of the project. It demonstrates how the track path actively improves and smooths out lap-after-lap without relying on AI or predefined track sectors.

### 4. `VelocityProfileManager.py` (Speed Optimization)
The speed control counterpart to the geometric learner.
* **Functionality:** Dynamically calculates the maximum safe velocity for every point on the track based on the newly generated candidate line's true geometric curvature. It automatically propagates braking zones backward from sharp turns.
* **Significance:** Validates the project's progress by demonstrating mathematically sound reductions in lap times as the racing line evolves.

## Author
**Stasinos Georgios (Στασινός Γεώργιος)** Electrical Engineering / F1TENTH Autonomous Racing Project
