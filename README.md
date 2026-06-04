# Iterative Racing-Line Learning MPC for F1TENTH

[![F1TENTH Simulation Video](https://img.youtube.com/vi/PZtI_nIbe9g/maxresdefault.jpg)](https://www.youtube.com/watch?v=PZtI_nIbe9g)
*Click the image above to view the simulation progress video on YouTube.*

## Project Overview
This repository contains the implementation of an Iterative Racing-Line Learning Model Predictive Controller (MPC) for the F1TENTH platform. The project is developed in **ROS 2 Foxy** using **Python** and relies on the **OSQP** solver for real-time optimization. 

A core philosophy of this project is the strict adherence to a **purely mathematical and iterative approach**. The learning mechanism relies entirely on global continuous optimization and dynamic curvature calculations, explicitly avoiding the use of Artificial Intelligence (AI) or localized track-sector heuristics. 

Furthermore, the controller is designed with advanced soft constraints (slack variables) that allow the vehicle to safely touch or graze track boundaries without the solver failing (becoming infeasible), promoting aggressive and continuous learning.

## Repository Structure & Development Methodology

To navigate this repository effectively, please focus on four critical Python scripts within the `mpc_controller` package. 

The development methodology for this project was highly modular. The three core components (MPC, Racing Line Manager, and Velocity Profile Manager) were first written, isolated, and tuned in their own standalone files to ensure mathematical stability before being integrated into the final master node.

### 1. `mpc_controller.py` (The Final Integrated Node)
This is the main executable file that combines all three classes and actually drives the vehicle in the simulation. 
* **Functionality:** It runs the 10Hz real-time control loop, linearizes the Discrete Kinematic Bicycle Model, assembles the sparse QP matrices, and manages the lap-to-lap iterative learning updates. 
* **Significance:** This file represents the culmination of the entire project.

### 2. `MPC.py` (Isolated Tuning Class)
* **Functionality:** Contains just the base MPC mathematical model.
* **Significance:** This file was extracted to tune the controller independently. It was used to adjust the weight matrices (Q, R) and slack variables, ensuring the car could stably follow a basic, static reference line before introducing the complexity of the iterative learning algorithms.

### 3. `RacingLineManager.py` (Isolated Tuning Class)
* **Functionality:** Contains the geometric path-learning engine.
* **Significance:** Isolated to verify that the track-centric ILC approach works. It allowed for the independent testing of line mutations toward the apexes and confirmed that the Minimum Curvature QP smoothing generates physically viable, improved candidate lines lap after lap.

### 4. `VelocityProfileManager.py` (Isolated Tuning Class)
* **Functionality:** Contains the dynamic speed and braking calculations.
* **Significance:** Isolated to prove that the dynamic curvature math accurately generates safe, optimized velocity profiles. This component was tested separately to ensure it properly propagates braking zones and pushes the car to faster lap times as the racing line evolves.

## Author
**Stasinos Georgios (Στασινός Γεώργιος)** Electrical Engineering / F1TENTH Autonomous Racing Project
