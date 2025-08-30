import pybullet as p
import pybullet_data
import numpy as np
import math
import time

class RobotController:
    def __init__(self):
        self.setup_physics_world()
        self.load_robot()
        self.setup_debug_parameters()
        
    def setup_physics_world(self):
        self.physics_client = p.connect(p.GUI, options='--background_color_red=0.8 --background_color_green=1 --background_color_blue=0.9')
        p.setGravity(0, 0, -9.8)
        
        p.resetDebugVisualizerCamera(
            cameraDistance=2,
            cameraYaw=0,
            cameraPitch=-30,
            cameraTargetPosition=(0, 0, 0.55)
        )
        
        p.setRealTimeSimulation(1)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf")
        
    def load_robot(self):
        self.start_pos = [0, 0, 0.5]
        self.start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        
        try:
            self.robot_id = p.loadURDF(
                'mini_arm_URDF_V14/urdf/mini_arm_URDF_V14.urdf',
                self.start_pos,
                self.start_orientation,
                useFixedBase=True
            )
        except Exception as e:
            print(f"Failed to load robot model: {e}")
            return
        
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_info = []
        self.initial_joint_positions = []
        
        print(f"Robot has {self.num_joints} joints:")
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode('utf-8')
            joint_type = info[2]
            
            if joint_type != p.JOINT_FIXED:
                self.joint_info.append({
                    'index': i,
                    'name': joint_name,
                    'type': joint_type,
                    'lower_limit': info[8],
                    'upper_limit': info[9]
                })
                
                current_pos = p.getJointState(self.robot_id, i)[0]
                self.initial_joint_positions.append(current_pos)
                
                print(f"Joint {i}: {joint_name}, Type: {joint_type}, Limits: [{info[8]:.2f}, {info[9]:.2f}]")
    
    def setup_debug_parameters(self):
        self.joint_params = []
        self.reset_buttons = []
        self.last_button_values = []
        
        for i, joint in enumerate(self.joint_info):
            param_name = f"Joint_{joint['index']}_{joint['name']}"
            joint_param = p.addUserDebugParameter(
                param_name,
                joint['lower_limit'] if joint['lower_limit'] > -1000 else -math.pi,
                joint['upper_limit'] if joint['upper_limit'] < 1000 else math.pi,
                self.initial_joint_positions[i]
            )
            self.joint_params.append(joint_param)
            
            reset_btn = p.addUserDebugParameter(f"Reset_Joint_{joint['index']}", 1, 0, 0)
            self.reset_buttons.append(reset_btn)
            self.last_button_values.append(0)
        
        self.reset_all_btn = p.addUserDebugParameter("Reset_All_Joints", 1, 0, 0)
        self.last_reset_all_value = 0
    
    def set_joint_position(self, joint_index, target_position, max_force=100):
        p.setJointMotorControl2(
            bodyUniqueId=self.robot_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_position,
            force=max_force
        )
    
    def reset_joint_to_initial(self, joint_index):
        if joint_index < len(self.initial_joint_positions):
            initial_pos = self.initial_joint_positions[joint_index]
            self.set_joint_position(joint_index, initial_pos)
            print(f"Joint {joint_index} reset to initial position: {initial_pos:.3f}")
    
    def reset_all_joints(self):
        for i, joint in enumerate(self.joint_info):
            self.reset_joint_to_initial(joint['index'])
        
        p.resetBasePositionAndOrientation(self.robot_id, self.start_pos, self.start_orientation)
        print("All joints reset to initial state")
    
    def update_joint_controls(self):
        for i, joint in enumerate(self.joint_info):
            if i < len(self.joint_params):
                target_position = p.readUserDebugParameter(self.joint_params[i])
                self.set_joint_position(joint['index'], target_position)
        
        for i, reset_btn in enumerate(self.reset_buttons):
            current_value = p.readUserDebugParameter(reset_btn)
            if current_value != self.last_button_values[i]:
                if i < len(self.joint_info):
                    self.reset_joint_to_initial(self.joint_info[i]['index'])
                self.last_button_values[i] = current_value
        
        current_reset_all = p.readUserDebugParameter(self.reset_all_btn)
        if current_reset_all != self.last_reset_all_value:
            self.reset_all_joints()
            self.last_reset_all_value = current_reset_all
    
    def run_simulation(self):
        print("Starting simulation, use sliders to control robot joint positions")
        print("Press 'q' or close window to exit simulation")
        
        simulation_step_counter = 0
        
        try:
            while True:
                self.update_joint_controls()
                
                simulation_step_counter += 1
                if simulation_step_counter >= 10:
                    p.stepSimulation()
                    simulation_step_counter = 0
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("Simulation interrupted by user")
        except Exception as e:
            print(f"Error during simulation: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        if hasattr(self, 'physics_client'):
            p.disconnect(self.physics_client)
        print("Simulation ended, resources cleaned up")

def main():
    try:
        controller = RobotController()
        controller.run_simulation()
    except Exception as e:
        print(f"Initialization failed: {e}")
        print("Please check if URDF file path is correct")

if __name__ == "__main__":
    main()