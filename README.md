## Mujoco_drone_landing


# Drone landing on moving target
 

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
    (Training v9) python PPO_train.py
    (Training v8) python PPO_train_custum-v2.py
    (Test) python PPO_check.py
    
    
##
### 1. Problem Statement

Our main Our fundamental goal is to control the robot in simulation using a vision-based RL algorithm.

TASK : Target tracking Locomotion

![image](https://user-images.githubusercontent.com/74540268/183004358-ea2d3f36-fce0-4717-adcd-ef64bc1f3c92.png)



##
### 2. Hardware 3D design

![hexy_heat2](https://user-images.githubusercontent.com/74540268/169944721-46a89900-eaed-4b17-b6cb-a4496fd48ab6.PNG)

URDF, xml : All links and joints are manually reverse engineered using assembly file from [Arcbotics](http://arcbotics.com/products/hexy/) 

##
### 3. MUJOCO camera sensor

![image](https://user-images.githubusercontent.com/74540268/183004430-8e820044-b7ad-48bb-8a43-1c519efd3879.png)


For mounting Camera on Robot Model, you can see the file in gym/mujoco/assets/Hexy_ver_2.3/assets

To get RGB data from camera for observation, you can see the file in gym/mujoco/hexy_v8.py


##
### 4. Policy Net

![image](https://user-images.githubusercontent.com/74540268/179349101-6eb8b4ff-d24e-486e-99dd-2e28ca9d6620.png)


• Input : Image RGB data + current motor angle


• Output : Desired motor angle

• RL algorithm : PPO


##
### 5. Training env
you can see the code for specific MDP setting(S,A,R..) info

    YOUR_PATH/python3.X/site-packages/gym/envs/mujoco/

![image](https://user-images.githubusercontent.com/74540268/183005146-f607a6bd-2653-4a20-bad5-2e9ca2ad9132.png)


**hexy_v8.py** : *TASK*

**hexy_v9.py** : *Simplified vector verification*


 
##
### 6. Results - Simplified vector obs verification


![image](https://user-images.githubusercontent.com/74540268/183004580-cc32688b-8a1c-4dca-adc0-e95dfffaec85.png)


##
### 7. Results - Camera obs + CNN feature extraction
  

![image](https://user-images.githubusercontent.com/74540268/183004664-03896098-c707-4258-b5c3-d086483bebf1.png)


##

### 8. VIDEO

**Video : [YOUTUBE link](https://youtu.be/n8gWz1U0qKk)**

You can watch **video** through the link above!



##
