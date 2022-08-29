### Autonomous Landing of Quadrotor on a Moving Target Using Vision-based RL
##

### 0. Requirements
 Installation : GYM, MUJOCO, Stable-Baselines3 + (Linux)

  1. Move to :

    YOUR_PATH/python3.X/site-packages/gym/

  2. Clone this repository in

    YOUR_PATH/python3.X/site-packages/gym/envs/
    
  3. Move [train_test_] directory to :

    YOUR_PATH/python3.X/site-packages/gym/
    
  4. Train/Test

    cd YOUR_PATH/python3.X/site-packages/gym/train_test_
    (Training v1) python PPO_train.py
    (Training v2) python PPO_train_custum-v3.py
    (Test) python PPO_check.py
    
    
##
### 1. Problem Statement

Our main Our fundamental goal is to control the robot in simulation using a vision-based RL algorithm.

TASK : Autonomous Quadrotor landing on a moving platform

![image](https://user-images.githubusercontent.com/74540268/185775849-b4881703-71ee-4586-89ba-700f46be91fe.png)

Control Input(=Neural network output) and Constraints

![image](https://user-images.githubusercontent.com/74540268/185776322-f5fa4236-0a05-459e-a7a0-0ec807ba0804.png)



##
### 2. Hardware 3D design (Quadrotor , Car)
Design Tool : Solidworks (+ SWtoURDF)

![image](https://user-images.githubusercontent.com/74540268/185776303-fdaf7820-8d68-4629-a288-89ec259749b0.png)

![image](https://user-images.githubusercontent.com/74540268/185776307-e8b13675-5676-4957-b6d4-6578ada9bd79.png)




##
### 3. MUJOCO camera sensor

![image](https://user-images.githubusercontent.com/74540268/185775919-4f4f988f-cbda-4e34-bc03-3f3266dd3ffe.png)

For mounting Camera on Robot Model, you can see the file in gym/mujoco/assets/Drone_ver_1.0/assets

To get RGB data from camera for observation, you can see the file in gym/mujoco/drone_v2.py


##
### 4. Policy Net

![image](https://user-images.githubusercontent.com/74540268/185775942-a8166c51-4d8f-469e-8736-09e43c570ecc.png)


• Input : Image RGB data + Current Action and Pitch angle [Vx, Vy, Vz, Pitch angle] 

• Output : Vx, Vy, Vz, Wy 

• RL algorithm : PPO


##
### 5. Training env
you can see the code for specific MDP setting(S,A,R..) info

    YOUR_PATH/python3.X/site-packages/gym/envs/mujoco/

![image](https://user-images.githubusercontent.com/74540268/185776069-f8405f81-4210-43bd-ae8e-ef32d35d75a7.png)

![image](https://user-images.githubusercontent.com/74540268/185776080-080c067f-6dfe-445c-acf4-8d8806997694.png)


**drone_v1.py** : *Simplified vector verification*

**drone_v2.py** : *TASK*



##
### 6. Turn-Off Flag
If the drone's **four points** touch the **landing box**, the propeller no longer needs to rotate
- So, Let’s set all control inputs to **zero**. (= Turn off)
- This algorithm is implemented using the **Turn-off flag.**

![image](https://user-images.githubusercontent.com/74540268/185776157-647001d5-972e-439f-9519-dd259a906a2a.png)



##
### 7. Results - Simplified vector obs verification


![image](https://user-images.githubusercontent.com/74540268/185776165-aca0bcc1-1f3d-4afd-a6a5-cca552157b31.png)

##
### 8. Results - Camera obs + CNN feature extraction
  

![image](https://user-images.githubusercontent.com/74540268/185776173-31debad8-05aa-4546-94ab-1093d1aadc95.png)


##

### 9. VIDEO

**Video : [YOUTUBE link](https://www.youtube.com/watch?v=5fczajDC63E)**

You can watch **video** through the link above.



##
