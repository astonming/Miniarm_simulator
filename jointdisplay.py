import pybullet as p
import pybullet_data
import numpy as np
import math
import time

import tkinter as tk


#p.GUI : 圖形化界面

physicsClient = p.connect(p.GUI, options='--background_color_red=0.8 --background_color_green=1 --background_color_blue=0.9')#設定物理引擎 # Connect to pybullet GUI
p.setGravity(0, 0, -9.8) #設定重力加速度 (gravity force along the X world axis,gravity force along the Y world axis,gravity force along the Z world axis)
p.resetDebugVisualizerCamera(
                            cameraDistance=2,            # 相機與目標位置之間的距離(m)
                            cameraYaw=0, # 水平旋轉角度,0 度代表正視，90 度表示從右側觀察，-90 度則表示從左側觀察,角度單位(度)
                            cameraPitch=-30, # 垂直旋轉角度 0 度表示水平視角，負值表示向下看，正值表示向上看 角度單位
                            cameraTargetPosition=(0, 0, 0.55) # 相機瞄準的位置
                            )
p.setRealTimeSimulation(1)# 設定模擬
# 添加資源路徑，內建urdf檔案的路徑添加進來
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")

#mugId = p.loadURDF("urdf/mug.urdf",basePosition = [1,0,0], baseOrientation = p.getQuaternionFromEuler([0, 0, 0]), globalScaling = 1, useFixedBase=True)#比例:globalScaling

StartPos = [0, 0, 0.5]
StartOrientation = p.getQuaternionFromEuler([0, 0, 0])
# RobotId = p.loadURDF('mini_arm_URDF_V10/urdf/mini_arm_URDF_V10.urdf', StartPos, StartOrientation,useFixedBase=True)
RobotId = p.loadURDF('mini_arm_URDF_V14/urdf/mini_arm_URDF_V14.urdf', StartPos, StartOrientation,useFixedBase=True)

# 取得機器人目前的關節位置
number_of_joints = p.getNumJoints(RobotId)
current_positions = [p.getJointState(RobotId, i)[0] for i in range(number_of_joints)]
linkid = [0] * number_of_joints
for joint_number in range(number_of_joints):
    info = p.getJointInfo(RobotId, joint_number)
    
    print(info[0], ": ", info[1], info[3])
    linkid[joint_number]=info[3]
    print(info)
def set_motor(jointID,joint_value):
    
    #print(f'Setting joint {jointID} to {joint_value}/n')
    p.setJointMotorControl2(
        bodyUniqueId=RobotId,     # 機器人的 ID
        jointIndex=jointID,             # 要控制的關節索引
        controlMode=p.POSITION_CONTROL, # 控制模式為位置控制
        targetPosition=joint_value,       # 目標位置 (弧度)
        force=100                 # 最大作用力
    )
   
joint_params = [p.addUserDebugParameter(f'j{i+1}', -math.pi, math.pi, 0) for i in range(number_of_joints)]
btn_params = [p.addUserDebugParameter(f'btn{i+1}', 1, 0, 0) for i in range(number_of_joints)]
btn_values = [p.readUserDebugParameter(btn) for btn in btn_params]
threshold_distance=0.01
while True:

    for i in range(number_of_joints):
        joint_value = p.readUserDebugParameter(joint_params[i])
        #print(f'joint{i+1} value: {joint_value}\n')
        set_motor(i, joint_value)
        for _ in range(10):  # 多次调用仿真步骤
            p.stepSimulation()

        # 检查按钮状态并重置机器人
        current_btn_value = p.readUserDebugParameter(btn_params[i])
        if current_btn_value != btn_values[i]: #如果沒有如預期到達一樣數值，按下按鈕會執行回到初始位置與角度
            p.resetBasePositionAndOrientation(RobotId, StartPos, StartOrientation)
            btn_values[i] = current_btn_value  # 更新按钮值
       
        