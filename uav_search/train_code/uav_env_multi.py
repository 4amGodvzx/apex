import gymnasium as gym
from gymnasium import spaces
import numpy as np
import airsim
import time
import json
import subprocess
import os
import signal
import logging
import datetime

from uav_search.airsim_utils import get_train_images
from uav_search.action_model_inputs_test import map_input_preparation
from uav_search.train_code.map_updating_train import map_update_simple



class AirSimDroneEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, worker_index, base_port=41451):
        super(AirSimDroneEnv, self).__init__()

        self.action_space = spaces.Discrete(6) 
        self.action = [0,1,2,3,4,5] # 0:forward, 1:rotate left, 2:rotate right, 3:rotate backward, 4:ascend, 5:descend
        self.observation_space = spaces.Dict({
            "attraction_map_input": spaces.Box(low=0, high=1, shape=(10, 20, 20), dtype=np.float32),
            "exploration_map_input": spaces.Box(low=0, high=1000, shape=(10, 20, 20), dtype=np.float32),
            "obstacle_map_input": spaces.Box(low=0, high=1, shape=(4, 8, 8), dtype=np.float32)
        })
        self.worker_index = worker_index
        self.api_port = base_port + worker_index * 1
        self.settings_path = self._generate_settings_json()
        self.client = None
        self.airsim_process = None
        self.task_data = json.load(open('uav_search/task_map/rl_tasks.json'))
        self.map_scripts = {
            "BrushifyUrban": "BrushifyUrban/BrushifyUrban.sh",
            "CabinLake": "CabinLake/CabinLake.sh",
            "DownTown": "DownTown/DownTown_test1.sh",
            "Neighborhood": "Neighborhood/NewNeighborhood.sh",
            "Slum": "Slum/Slum_test1.sh",
            "UrbanJapan": "UrbanJapan/UrbanJapan.sh",
            "Venice": "Venice/Vinice_test1.sh",
            "WesternTown": "WesternTown/WesternTown_test1.sh",
            "WinterTown": "WinterTown/WinterTown_test1.sh"
        }
        match self.worker_index:
            case 0:
                self.task_id = 0
            case 1:
                self.task_id = 4
            case 2:
                self.task_id = 8
            case 3:
                self.task_id = 12
            case 4:
                self.task_id = 16
            case 5:
                self.task_id = 20
            case 6:
                self.task_id = 24
            case 7:
                self.task_id = 28
            case _:
                self.task_id = 0
        self.episode_id = 0
        self.current_map_name = None
        self.start_position = airsim.Vector3r(0,0,0)
        self.start_orientation = airsim.utils.to_quaternion(0, 0, 0)
        self.target_position = np.array([0.0, 0.0, 0.0])
        self.episode_step_count = 0
        self.max_steps_per_episode = 100
        # Map
        self._map_reset()
        self.grid_size = 5.0
        self.grid_origin_pose = np.array([20, 20, 5])
        # Distance to target
        self.last_dist_to_target = 0.0
        self.current_dist_to_target = 0.0
        # Log info
        self.log = {
            "ep_distance": 0.0,
            "ep_step": 0.0,
            "ep_reward_distance": 0.0,
            "ep_reward_sparse": 0.0,
            "ep_reward_attraction": 0.0,
            "ep_reward_exploration": 0.0,
            "ep_forward": 0.0
        }
        self.forward = 0.0
        self.reward_log = [0,0,0,0]
        self.distance_sum = 0.0
        print("AirSim environment initialized.")
    
    def _generate_settings_json(self):
        settings_dict = {
            "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/main/docs/settings.md",
            "SettingsVersion": 1.2,
            "SimMode": "Multirotor",
            "ClockSpeed": 4.0,
            "ApiServerPort": self.api_port,
            "CameraDefaults": {
                "CaptureSettings": [
                    {
                        "ImageType": 0,
                        "Width": 512,
                        "Height": 512,
                        "FOV_Degrees": 90,
                        "AutoExposureSpeed": 100,
                        "AutoExposureMaxBrightness": 0.64,
                        "AutoExposureMinBrightness": 0.03
                    },
                    {
                        "ImageType": 1,
                        "Width": 64,
                        "Height": 64,
                        "FOV_Degrees": 90,
                        "AutoExposureSpeed": 100,
                        "AutoExposureMaxBrightness": 0.64,
                        "AutoExposureMinBrightness": 0.03
                    },
                    {
                        "ImageType": 2,
                        "Width": 64,
                        "Height": 64,
                        "FOV_Degrees": 90,
                        "AutoExposureSpeed": 100,
                        "AutoExposureMaxBrightness": 0.64,
                        "AutoExposureMinBrightness": 0.03
                    }
                ]
            }
        }

        settings_filename = f"settings_{self.worker_index}.json"
        temp_dir = "airsim_settings"
        os.makedirs(temp_dir, exist_ok=True)
        settings_path = os.path.join(temp_dir, settings_filename)
        
        with open(settings_path, 'w') as f:
            json.dump(settings_dict, f, indent=4)
        
        return os.path.abspath(settings_path)
    
    def _launch_or_switch_map(self, target_map_name):
        if target_map_name == self.current_map_name and self.airsim_process and self.airsim_process.poll() is None:
            print(f"Map '{target_map_name}' is already running.")
            return True
        print(f"Switching map... Current: '{self.current_map_name}', Target: '{target_map_name}'")

        if self.airsim_process:
            print("Terminating existing AirSim process...")
            self.close()

        script_path = self.map_scripts.get(target_map_name)
        if not script_path:
            raise ValueError(f"No launch script found for map: {target_map_name}")
        
        print(f"Launching new AirSim process with script: {script_path}")
        launch_command = ['bash', script_path, '-RenderOffscreen', '-NoSound', '-NoVSync', '-GraphicsAdapter=6', f'-settings="{self.settings_path}"']
        self.airsim_process = subprocess.Popen(launch_command,start_new_session=True)
        self.current_map_name = target_map_name
        self._connect_to_airsim()
    
    def _connect_to_airsim(self):
        print("Attempting to connect to AirSim...")
        time.sleep(10)
        while True:
            try:
                self.client = airsim.MultirotorClient(port=self.api_port)
                self.client.confirmConnection()
                print("Successfully connected to AirSim!")
                break
            except Exception as e:
                print(f"Connection failed: {e}. Retrying in 3 seconds...")
                time.sleep(3)
    
    def _map_reset(self):
        self.current_ground_truth_attraction_map = np.loadtxt(f'uav_search/task_map/task_{self.task_id}.txt').reshape((40, 40, 10))
        self.attraction_map = np.zeros((40, 40, 10, 2), dtype=np.float32)
        self.attraction_map[:, :, :, 1] = -1
        self.exploration_map = np.zeros((40, 40, 10), dtype=np.float32)
        self.obstacle_map = np.zeros((40, 40, 10), dtype=np.float32)
        self.uav_pose = {
            'position': np.array([20, 20, 5]),
            'orientation': 0 # 0: north, 1: west, 2: south, 3: east
            }
            
    def _compute_new_position(self, forward_distance, position: airsim.Vector3r, yaw) -> airsim.Vector3r:
        dx = forward_distance * np.cos(yaw)
        dy = forward_distance * np.sin(yaw)

        new_position = airsim.Vector3r(position.x_val + dx, position.y_val + dy, position.z_val)
        return new_position
    
    def _map_update(self):
        depth_image, camera_position, camera_orientation = get_train_images(self.client)
        camera_fov = 90

        new_attraction_map, new_exploration_map, new_obstacle_map, attraction_reward, exploration_reward = map_update_simple(
            self.attraction_map,
            self.exploration_map,
            self.obstacle_map,
            self.current_ground_truth_attraction_map,
            self.start_position,
            depth_image,
            camera_fov,
            camera_position,
            camera_orientation,
            self.uav_pose["orientation"]
        )
            
        self.attraction_map = new_attraction_map
        self.exploration_map = new_exploration_map
        self.obstacle_map = new_obstacle_map
        
        return attraction_reward, exploration_reward

    def _update_uav_pose_from_airsim(self):
        state = self.client.getMultirotorState()
        position = state.kinematics_estimated.position
        orientation = state.kinematics_estimated.orientation

        delta_position = np.array([
            position.x_val - self.start_position.x_val,
            position.y_val - self.start_position.y_val,
            position.z_val - self.start_position.z_val 
        ])
        
        grid_position = np.round(delta_position / self.grid_size).astype(int) + self.grid_origin_pose

        _, _, yaw_rad = airsim.to_eularian_angles(orientation)
        yaw_deg = np.degrees(yaw_rad)
        
        if -45 <= yaw_deg <= 45:
            grid_orientation = 0 # north
        elif -135 < yaw_deg <= -45:
            grid_orientation = 1 # west
        elif yaw_deg > 135 or yaw_deg < -135:
            grid_orientation = 2 # south
        else:
            grid_orientation = 3 # east
            
        self.uav_pose['position'] = grid_position
        self.uav_pose['orientation'] = grid_orientation

    def _get_obs(self, action, attraction_reward, exploration_reward):
        map_input = map_input_preparation(self.attraction_map, self.exploration_map, self.obstacle_map, self.uav_pose) 
        return map_input

    def _compute_reward(self,terminated, attraction_reward=0.0, exploration_reward=0.0):
        # Reward weights
        W_ATTRACTION = 1.0
        W_EXPLORATION = 0.5
        W_DISTANCE = 0.0
        W_SPARSE = 1.0
        STEP_PENALTY = 0.0
        
        # Distance-based reward
        distance_decrease = self.last_dist_to_target - self.current_dist_to_target
        if 10 <= self.current_dist_to_target < 30:
            k = 2.0
        elif 30 <= self.current_dist_to_target < 80:
            k = 1.0
        elif 80 <= self.current_dist_to_target < 150:
            k = 0.5
        else: # >= 150
            k = 0.2
        
        dis_reward = k * distance_decrease
        
        sparse_reward = 0.0
        if terminated:
            if self.current_dist_to_target < 10.0: # successful termination
                sparse_reward = 50.0
            else: # failure termination
                sparse_reward = -50.0
            
        # Log
        self.reward_log[0] += W_DISTANCE * dis_reward
        self.reward_log[1] += W_SPARSE * sparse_reward
        self.reward_log[2] += W_ATTRACTION * attraction_reward
        self.reward_log[3] += W_EXPLORATION * exploration_reward

        return W_DISTANCE * dis_reward + W_SPARSE * sparse_reward + W_ATTRACTION * attraction_reward + W_EXPLORATION * exploration_reward + STEP_PENALTY

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        print(f"Resetting environment for task {self.task_id}")
        selected_task = self.task_data[self.task_id]
        
        self._launch_or_switch_map(selected_task['map'])
        
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        
        start_pose_list = selected_task['start_position']
        self.start_position = airsim.Vector3r(start_pose_list[0], start_pose_list[1], start_pose_list[2]) # From dataset
        self.start_orientation = airsim.utils.to_quaternion(0, 0, 0)
        pose = airsim.Pose(self.start_position, self.start_orientation)
        self.client.simSetVehiclePose(pose, True)
        
        target_pose_list = selected_task['object_position']
        self.target_position = np.array(target_pose_list) # From dataset
        self.current_dist_to_target = np.linalg.norm(self.target_position - np.array([self.start_position.x_val, self.start_position.y_val, self.start_position.z_val]))
        self.last_dist_to_target = self.current_dist_to_target

        self.episode_step_count = 0
        
        # Map reset
        self._map_reset()

        _, _ = self._map_update()
        initial_observation = self._get_obs(-1, 0, 0)
        
        # Log info
        log_dir = "uav_search/train_logs"
        os.makedirs(log_dir, exist_ok=True)
        log_filename = os.path.join(log_dir, f"worker_{self.worker_index}_reset_log.txt")
        with open(log_filename, 'a') as log_file:
            log_file.write(f"{datetime.datetime.now()} - Task {self.task_id}, step_cnt: {self.log['ep_step']}, dis_mean: {self.log['ep_distance']}, reward_d_mean: {self.log['ep_reward_distance']}, reward_s: {self.log['ep_reward_sparse']}, reward_a_mean: {self.log['ep_reward_attraction']}, reward_e_mean: {self.log['ep_reward_exploration']}, forward: {self.log['ep_forward']}\n")
        self.reward_log = [0,0,0,0]
        self.distance_sum = 0.0
        self.log = {
            "ep_distance": 0.0,
            "ep_step": 0.0,
            "ep_reward_distance": 0.0,
            "ep_reward_sparse": 0.0,
            "ep_reward_attraction": 0.0,
            "ep_reward_exploration": 0.0,
            "ep_forward": 0.0
        }
        self.forward = 0.0
        info = self.log

        self.episode_id += 1
        if self.episode_id >= 10:
            self.episode_id = 0
            self.task_id = (self.task_id + 1) % len(self.task_data)

        return initial_observation, info

    def step(self, action):
        try:
            self.episode_step_count += 1
            
            state = self.client.getMultirotorState()
            position = state.kinematics_estimated.position
            orientation = state.kinematics_estimated.orientation
            _, _, yaw = airsim.to_eularian_angles(orientation)
            
            np_position = np.array([position.x_val, position.y_val, position.z_val])
            new_position = self._compute_new_position(10, position, yaw)
            
            YAW_ANGLES = [0, -90, 180, 90]
            
            match action:
                case 0: 
                    self.client.moveToPositionAsync(new_position.x_val, new_position.y_val, new_position.z_val, 3, timeout_sec=5).join()
                    self.forward += 1.0
                case 1:
                    current_orientation_idx = self.uav_pose['orientation']
                    new_orientation_idx = (current_orientation_idx + 1) % 4
                    target_yaw = YAW_ANGLES[new_orientation_idx]
                    self.client.rotateToYawAsync(target_yaw, timeout_sec=3).join()
                case 2:
                    current_orientation_idx = self.uav_pose['orientation']
                    new_orientation_idx = (current_orientation_idx - 1 + 4) % 4
                    target_yaw = YAW_ANGLES[new_orientation_idx]
                    self.client.rotateToYawAsync(target_yaw, timeout_sec=3).join()
                case 3:
                    current_orientation_idx = self.uav_pose['orientation']
                    new_orientation_idx = (current_orientation_idx + 2) % 4
                    target_yaw = YAW_ANGLES[new_orientation_idx]
                    self.client.rotateToYawAsync(target_yaw, timeout_sec=2).join()
                case 4:
                    self.client.moveToZAsync(position.z_val - self.grid_size, 2, timeout_sec=3).join()
                case 5:
                    self.client.moveToZAsync(position.z_val + self.grid_size, 2, timeout_sec=3).join()
            
            self._update_uav_pose_from_airsim()
            
            self.distance_sum += self.current_dist_to_target
            
            terminated = False
            self.last_dist_to_target = self.current_dist_to_target
            self.current_dist_to_target = np.linalg.norm(self.target_position - np_position)
            collision_info = self.client.simGetCollisionInfo()

            if self.uav_pose['position'][0] < 0 or self.uav_pose['position'][0] >= 40 or \
            self.uav_pose['position'][1] < 0 or self.uav_pose['position'][1] >= 40 or \
            self.uav_pose['position'][2] < 0 or self.uav_pose['position'][2] >= 10:
                terminated = True
                print("Out of bounds! Episode terminated.")

            if collision_info.has_collided:
                terminated = True
                print("Collision detected! Episode terminated.")
                self.client.reset()
            
            if self.current_dist_to_target < 10.0:
                terminated = True
                print("Target reached! Episode terminated.")

            truncated = False
            if self.episode_step_count >= self.max_steps_per_episode:
                truncated = True
                print("Max steps reached. Episode truncated.")
            
            if not terminated and not truncated:
                attraction_reward, exploration_reward = self._map_update()
                observation = self._get_obs(action, attraction_reward, exploration_reward)
                reward = self._compute_reward(terminated, attraction_reward, exploration_reward)
                info = {}
            else:
                attraction_reward, exploration_reward = 0.0, 0.0
                observation = self._get_obs(action, attraction_reward, exploration_reward)
                reward = self._compute_reward(terminated, attraction_reward, exploration_reward)
                # Log info
                self.log = {
                    "ep_step": self.episode_step_count,
                    "ep_distance": self.distance_sum / self.episode_step_count,
                    "ep_reward_distance": self.reward_log[0] / self.episode_step_count,
                    "ep_reward_sparse": self.reward_log[1],
                    "ep_reward_attraction": self.reward_log[2] / self.episode_step_count,
                    "ep_reward_exploration": self.reward_log[3] / self.episode_step_count,
                    "ep_forward": self.forward
                }
                info = self.log
        
        except Exception as e:
            print(f"An error occurred during step execution: {e}")
            observation = self._get_obs(-1, 0, 0)
            reward = 0.0
            terminated = True
            truncated = False
            logging.basicConfig(filename='uav_search/logs/error.log', level=logging.ERROR)
            logging.error("Exception occurred", exc_info=True)
            info = {"error": str(e)}

        return observation, reward, terminated, truncated, info

    def close(self):
        if self.airsim_process and self.airsim_process.poll() is None:
            print(f"Attempting to terminate AirSim process group with PGID: {os.getpgid(self.airsim_process.pid)}")
            try:
                pgid = os.getpgid(self.airsim_process.pid)
                os.killpg(pgid, signal.SIGTERM)
                self.airsim_process.wait(timeout=10)
                print("Process group terminated gracefully.")
                time.sleep(2)
                os.killpg(pgid, signal.SIGTERM)
                time.sleep(1)
            except subprocess.TimeoutExpired:
                print("Process group did not terminate gracefully, forcing kill (SIGKILL).")
                pgid = os.getpgid(self.airsim_process.pid)
                os.killpg(pgid, signal.SIGKILL)
                self.airsim_process.wait()
                print("Process group killed.")
            except ProcessLookupError:
                print("Process was already gone before termination signal could be sent.")
            except Exception as e:
                print(f"An error occurred while closing the process group: {e}")
        self.airsim_process = None
        self.client = None
        time.sleep(7)
        print("Environment closed.")