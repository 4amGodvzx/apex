import multiprocessing
import subprocess
import time
import signal
import json
import datetime
import os
import airsim
import numpy as np
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from gymnasium import spaces

from uav_search.airsim_utils import get_images
from uav_search.grounded_sam_test import grounded_sam
from uav_search.map_updating_numpy import add_masks, downsample_masks, map_update_simple
from uav_search.detection_test import detection_test
from uav_search.action_model_inputs_test import obstacle_update, map_input_preparation

DEVICE = "cuda:3"
DINO_MODEL_DIR = "models/models-grounding-dino-base"
SAM_MODEL_DIR = "models/models-sam-vit-base"
ACTION_MODEL_PATH = "uav_search/models/f_ppo_num_4_final_400000.zip"
STATS_PATH = "uav_search/models/vec_normalize_f_ppo_num_4_final.pkl"

GRID_SCALE = 5.0
ATTRACTION_MAP_SIZE = (40, 40, 10)
OBSTACLE_MAP_SIZE = (40, 40, 10)
UAV_START_GRID_POS = np.array([20, 20, 5])

OBSERVATION_SPACE = spaces.Dict({
            "attraction_map_input": spaces.Box(low=0, high=1, shape=(10, 20, 20), dtype=np.float32),
            "exploration_map_input": spaces.Box(low=0, high=1000, shape=(10, 20, 20), dtype=np.float32),
            "obstacle_map_input": spaces.Box(low=0, high=1, shape=(4, 8, 8), dtype=np.float32)
        })
ACTION_SPACE = spaces.Discrete(6)

YAW_ANGLES = [0, -90, 180, 90]  # North, West, South, East

class MockVecEnv:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_envs = 1
        self.render_mode = None

    def step(self, actions):
        raise NotImplementedError


    def reset(self):
        raise NotImplementedError

    def close(self):
        pass

class UAVSearchAgent:
    def __init__(self, start_position: airsim.Vector3r, target_position: airsim.Vector3r, object_name: str, object_description: str, log_dir="experiment_logs"):
        self.start_position = start_position
        self.start_orientation = airsim.utils.to_quaternion(0, 0, 0)
        self.target_position = target_position
        self.object_name = object_name
        self.object_description = object_description

        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"experiment_{timestamp}.json")

        self.manager = multiprocessing.Manager()
        self._initialize_shared_memory()

    def _initialize_shared_memory(self):
        attraction_map = np.zeros((*ATTRACTION_MAP_SIZE, 2), dtype=np.float32)
        attraction_map[..., 1] = -1  # Initialize weights to -1
        exploration_map = np.zeros(ATTRACTION_MAP_SIZE, dtype=np.float32)
        obstacle_map = np.zeros(OBSTACLE_MAP_SIZE, dtype=np.float32)

        self.shared_maps = self.manager.dict({
            'attraction_map_buffer': attraction_map.tobytes(),
            'exploration_map_buffer': exploration_map.tobytes(),
            'obstacle_map_buffer': obstacle_map.tobytes(),
        })

        self.shared_detection_info = self.manager.dict({
            'detection_success': False,
            'detected_position': self.manager.list([0.0, 0.0, 0.0]),
            'reach_max_steps': False,
            'terminated': False
        })

        self.data_lock = self.manager.Lock()
        self.detection_lock = self.manager.Lock()

    def _update_uav_pose_from_airsim(self, state: airsim.MultirotorState):
        position = state.kinematics_estimated.position
        orientation = state.kinematics_estimated.orientation

        delta_position = np.array([
            position.x_val - self.start_position.x_val,
            position.y_val - self.start_position.y_val,
            position.z_val - self.start_position.z_val
        ])
        
        grid_position = np.round(delta_position / GRID_SCALE).astype(int) + UAV_START_GRID_POS

        _, _, yaw_rad = airsim.to_eularian_angles(orientation)
        yaw_deg = np.degrees(yaw_rad)
        
        if -45 <= yaw_deg <= 45:
            grid_orientation = 0  # north
        elif -135 < yaw_deg <= -45:
            grid_orientation = 1  # west
        elif yaw_deg > 135 or yaw_deg < -135:
            grid_orientation = 2  # south
        else:
            grid_orientation = 3  # east
            
        uav_pose = {'position': grid_position, 'orientation': grid_orientation}
        return uav_pose

    def _action_process(self):
        
        action_client = airsim.MultirotorClient(port=41460)
        action_client.confirmConnection()
        action_client.reset()
        action_client.enableApiControl(True)
        action_client.armDisarm(True)

        action_client.takeoffAsync().join()
        pose = airsim.Pose(self.start_position, self.start_orientation)
        action_client.simSetVehiclePose(pose, True)
        
        uav_pose = {'position': UAV_START_GRID_POS, 'orientation': 0}
        path_length = 0.0
        
        action_model = PPO.load(ACTION_MODEL_PATH, device=DEVICE)
        mock_env = MockVecEnv(OBSERVATION_SPACE, ACTION_SPACE)
        vec_normalize_object = VecNormalize.load(STATS_PATH, mock_env)
        vec_normalize_object.training = False
        vec_normalize_object.norm_reward = False
        
        is_success = False
        oracle_success_achieved = False
        termination_reason = "max_steps_reached"
        navigation_error = float('inf')

        start_pos_vec = np.array([self.start_position.x_val, self.start_position.y_val, self.start_position.z_val])
        target_pos_vec = np.array([self.target_position.x_val, self.target_position.y_val, self.target_position.z_val])
        optimal_path_length = np.linalg.norm(start_pos_vec - target_pos_vec)
        
        experiment_log = {
            "setup": {
                "start_position": [start_pos_vec[0],start_pos_vec[1],start_pos_vec[2]],
                "target_position": [target_pos_vec[0],target_pos_vec[1],target_pos_vec[2]],
                "object_name": self.object_name
            },
            "step_data": [],
            "episode_summary": {}
        }
        
        print("[Action] Initialize complete.")

        for i in range(200): # Max steps
            log_entry = {'step': i}

            with self.detection_lock:
                is_detected = self.shared_detection_info['detection_success']
            if is_detected:
                detected_pos_list = list(self.shared_detection_info['detected_position'])
                detected_pos_vec = np.array(detected_pos_list)
                print(f"[Action] Detection success! Moving to detected position: {detected_pos_list}")
                navigation_error = np.linalg.norm(detected_pos_vec - target_pos_vec)
                if navigation_error < 10:
                    is_success = True
                    termination_reason = "detection_success"
                else:
                    is_success = False
                    termination_reason = "detection_failure_inaccurate"

                action_client.moveToZAsync(-30, 2).join()
                action_client.moveToPositionAsync(detected_pos_list[0], detected_pos_list[1], -30, 5).join()
                path_length += np.linalg.norm(start_pos_vec - detected_pos_vec)
                print("[Action] Reached detected position.")
                break

            _, depth_image, camera_position, camera_orientation, _, _ = get_images(action_client)
            camera_fov = 90
            
            step_start_time = time.time()
            with self.data_lock:
                obstacle_map_copy = np.frombuffer(self.shared_maps['obstacle_map_buffer'], dtype=np.float32).reshape(OBSTACLE_MAP_SIZE)
            
            new_obstacle_map = obstacle_update(obstacle_map_copy, self.start_position, depth_image, camera_fov, camera_position, camera_orientation)
            
            with self.data_lock:
                self.shared_maps['obstacle_map_buffer'] = new_obstacle_map.tobytes()
                attraction_map_copy = np.frombuffer(self.shared_maps['attraction_map_buffer'], dtype=np.float32).reshape((*ATTRACTION_MAP_SIZE, 2))
                exploration_map_copy = np.frombuffer(self.shared_maps['exploration_map_buffer'], dtype=np.float32).reshape(ATTRACTION_MAP_SIZE)
            
            action_model_input = map_input_preparation(attraction_map_copy, exploration_map_copy, new_obstacle_map, uav_pose)
            normalized_obs = vec_normalize_object.normalize_obs(action_model_input)
            action, _ = action_model.predict(normalized_obs, deterministic=True)
            action = int(action)

            log_entry['step_duration'] = time.time() - step_start_time
            log_entry['uav_pose_before_action'] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in uav_pose.items()}
            log_entry['action_taken'] = action

            state = action_client.getMultirotorState()
            position = state.kinematics_estimated.position
            orientation = state.kinematics_estimated.orientation
            _, _, yaw = airsim.to_eularian_angles(orientation)
            
            dx = 10.0 * np.cos(yaw)
            dy = 10.0 * np.sin(yaw)
            new_position = airsim.Vector3r(position.x_val + dx, position.y_val + dy, position.z_val)
            
            if action == 0:
                action_client.moveToPositionAsync(new_position.x_val, new_position.y_val, new_position.z_val, 3, timeout_sec=5).join()
                path_length += 10.0
            elif action == 1:
                target_yaw = YAW_ANGLES[(uav_pose['orientation'] + 1) % 4]
                action_client.rotateToYawAsync(target_yaw, timeout_sec=3).join()
            elif action == 2:
                target_yaw = YAW_ANGLES[(uav_pose['orientation'] - 1 + 4) % 4]
                action_client.rotateToYawAsync(target_yaw, timeout_sec=3).join()
            elif action == 3:
                target_yaw = YAW_ANGLES[(uav_pose['orientation'] + 2) % 4]
                action_client.rotateToYawAsync(target_yaw, timeout_sec=3).join()
            elif action == 4:
                action_client.moveToZAsync(position.z_val - 5.0, 2, timeout_sec=3).join()
                path_length += 5.0
            elif action == 5:
                action_client.moveToZAsync(position.z_val + 5.0, 2, timeout_sec=3).join()
                path_length += 5.0

            print(f"[Action] Step {i} movement finished.")
            
            new_state = action_client.getMultirotorState()
            uav_pose = self._update_uav_pose_from_airsim(new_state)
            
            # Termination check
            should_terminate = False
            collision_info = action_client.simGetCollisionInfo()
            if collision_info.has_collided:
                should_terminate = True
                termination_reason = "collision"
            
            pos = uav_pose['position']
            if not (0 <= pos[0] < ATTRACTION_MAP_SIZE[0] and 0 <= pos[1] < ATTRACTION_MAP_SIZE[1] and 0 <= pos[2] < ATTRACTION_MAP_SIZE[2]):
                should_terminate = True
                termination_reason = "out_of_bounds"
                
            current_pos_vec = np.array([new_state.kinematics_estimated.position.x_val, new_state.kinematics_estimated.position.y_val, new_state.kinematics_estimated.position.z_val])
            dis_to_target_oracle = np.linalg.norm(current_pos_vec - target_pos_vec)
            if dis_to_target_oracle < 10:
                oracle_success_achieved = True

            log_entry['uav_pose_after_action'] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in uav_pose.items()}
            log_entry['distance_to_target_oracle'] = dis_to_target_oracle
            experiment_log["step_data"].append(log_entry)
            
            if should_terminate:
                print(f"[Action] Terminated due to: {termination_reason}")
                break

        print(f"[Action] Episode finished. Reason: {termination_reason}")

        if not is_success and navigation_error == float('inf'):
            final_state = action_client.getMultirotorState()
            final_pos_vec = np.array([final_state.kinematics_estimated.position.x_val, final_state.kinematics_estimated.position.y_val, final_state.kinematics_estimated.position.z_val])
            navigation_error = np.linalg.norm(final_pos_vec - target_pos_vec)

        spl = 0.0
        if is_success:
            spl = optimal_path_length / max(optimal_path_length, path_length) if path_length > 0 else 1.0

        experiment_log["episode_summary"] = {
            "success": is_success,
            "oracle_success": oracle_success_achieved,
            "termination_reason": termination_reason,
            "total_steps": i + 1,
            "path_length_actual": path_length,
            "path_length_optimal": optimal_path_length,
            "navigation_error": navigation_error,
            "spl": spl
        }

        with self.detection_lock:
            self.shared_detection_info['terminated'] = True
            if termination_reason == "max_steps_reached":
                self.shared_detection_info['reach_max_steps'] = True

        with open(self.log_file, 'w') as f:
            json.dump(experiment_log, f, indent=4)
        print(f"Log saved to {self.log_file}")

    def _detection_process(self):
        time.sleep(2)
        detection_dino_processor = AutoProcessor.from_pretrained(DINO_MODEL_DIR)
        detection_dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_DIR).to(DEVICE)
        print("[Detection] Models loaded.")

        detection_client = airsim.MultirotorClient(port=41460)
        detection_client.confirmConnection()
        k = 0

        while True:
            with self.detection_lock:
                if self.shared_detection_info['terminated']:
                    break
            
            pil_image, depth_image, camera_position, camera_orientation, _, _ = get_images(detection_client)
            camera_fov = 90

            point_world = detection_test(detection_dino_processor, detection_dino_model, pil_image, depth_image, camera_fov, camera_position, camera_orientation, self.object_name)
            
            if point_world is not None:
                with self.detection_lock:
                    if not self.shared_detection_info['detection_success']:
                        self.shared_detection_info['detection_success'] = True
                        self.shared_detection_info['detected_position'][:] = [float(p) for p in point_world]
                        print(f"[Detection] Target '{self.object_name}' found at {point_world}")

            print(f"[Detection] Cycle {k} Done.")
            time.sleep(2)
            k += 1
        print("[Detection] Process terminating.")

    def _planning_process(self):
        time.sleep(5) # Ensure AirSim is ready
        print("[Planning] Loading models...")
        dino_processor = AutoProcessor.from_pretrained(DINO_MODEL_DIR)
        dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_DIR).to(DEVICE)
        sam_processor = SamProcessor.from_pretrained(SAM_MODEL_DIR)
        sam_model = SamModel.from_pretrained(SAM_MODEL_DIR).to(DEVICE)
        print("[Planning] Models loaded.")

        planning_client = airsim.MultirotorClient(port=41460)
        planning_client.confirmConnection()
        j = 0

        while True:
            with self.detection_lock:
                if self.shared_detection_info['terminated'] or self.shared_detection_info['detection_success']:
                    break
            
            pil_image, depth_image, camera_position, camera_orientation, _, rgb_base64 = get_images(planning_client)
            camera_fov = 90
            
            result_dict, attraction_scores = grounded_sam(None, None, dino_processor, dino_model, sam_processor, sam_model, pil_image, rgb_base64, self.object_description)

            if not result_dict["success"]:
                print("[Planning] Grounded-SAM failed, skipping this frame.")
                time.sleep(1)
                continue

            with self.data_lock:
                attraction_map = np.frombuffer(self.shared_maps['attraction_map_buffer'], dtype=np.float32).reshape((*ATTRACTION_MAP_SIZE, 2))
                exploration_map = np.frombuffer(self.shared_maps['exploration_map_buffer'], dtype=np.float32).reshape(ATTRACTION_MAP_SIZE)

            added_masks = add_masks(result_dict["masks"])
            prepared_masks = downsample_masks(added_masks, scale_factor=2)
            state = planning_client.getMultirotorState()
            uav_pose = self._update_uav_pose_from_airsim(state)
            new_attraction_map, new_exploration_map, _ = map_update_simple(attraction_map, exploration_map, prepared_masks, attraction_scores, self.start_position, depth_image, camera_fov, camera_position, camera_orientation, uav_pose["orientation"])

            with self.data_lock:
                self.shared_maps['attraction_map_buffer'] = new_attraction_map.tobytes()
                self.shared_maps['exploration_map_buffer'] = new_exploration_map.tobytes()

            print(f"[Planning] Cycle {j} Done.")
            j += 1
        print("[Planning] Process terminating.")

    def run(self):
        processes = [
            multiprocessing.Process(target=self._action_process, name="Action"),
            multiprocessing.Process(target=self._planning_process, name="Planning"),
            multiprocessing.Process(target=self._detection_process, name="Detection")
        ]

        print("Starting processes...")
        for p in processes:
            p.start()

        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            print("\nCaught KeyboardInterrupt, terminating processes.")
            for p in processes:
                p.terminate()
            time.sleep(1) # Give time for processes to terminate
            for p in processes:
                p.join()
        finally:
            print("Resetting AirSim environment...")
            client = airsim.MultirotorClient(port=41460)
            client.reset()
            client.armDisarm(False)
            client.enableApiControl(False)
            print("Done.")

def wait_for_airsim_ready(timeout_sec=120):
    time.sleep(10)
    start_time = time.time()
    while time.time() - start_time < timeout_sec:
        try:
            print("Waiting for AirSim to become ready...")
            client = airsim.MultirotorClient(port=41460)
            client.confirmConnection()
            print("AirSim connection confirmed.")
            time.sleep(2)
            return True
        except:
            print("Connection failed, retrying in 5 seconds...")
            time.sleep(5)
    print("Error: AirSim did not become ready within the timeout period.")
    return False

class ExperimentRunner:
    def __init__(self, tasks_file, map_scripts_config, base_log_dir="experiment_logs"):
        self.tasks_file = tasks_file
        self.map_scripts = map_scripts_config
        self.base_log_dir = base_log_dir
        self.tasks = []
        self.current_map_name = None
        self.airsim_process = None
        os.makedirs(self.base_log_dir, exist_ok=True)

    def load_tasks(self):
        with open(self.tasks_file, 'r') as f:
            self.tasks = json.load(f)
        print(f"Successfully loaded {len(self.tasks)} tasks from {self.tasks_file}")

    def _manage_airsim_map(self, target_map_name):
        if target_map_name == self.current_map_name and self.airsim_process and self.airsim_process.poll() is None:
            print(f"Map '{target_map_name}' is already running.")
            return True

        print(f"Switching map... Current: '{self.current_map_name}', Target: '{target_map_name}'")

        if self.airsim_process:
            self.cleanup()

        script_path = self.map_scripts.get(target_map_name)
        if not script_path:
            print(f"Error: No launch script found for map: {target_map_name}")
            return False
        
        print(f"Launching new AirSim process for map '{target_map_name}'...")
        launch_command = ['bash', script_path, '-RenderOffscreen', '-NoSound', '-NoVSync', '-GraphicsAdapter=3']
        self.airsim_process = subprocess.Popen(launch_command, start_new_session=True)
        self.current_map_name = target_map_name
        
        return wait_for_airsim_ready()

    def run_all_experiments(self):
        self.load_tasks()
        
        try:
            for task in self.tasks:
                task_id = task['task_id']
                map_name = task['map']
                
                print("\n" + "="*50)
                print(f"STARTING TASK {task_id}: Find '{task['object_name']}' in map '{map_name}'")
                print("="*50)

                if not self._manage_airsim_map(map_name):
                    print(f"Failed to launch map {map_name}. Skipping task {task_id}.")
                    continue

                start_pos_list = task['start_position']
                target_pos_list = task['object_position']
                exp_start_position = airsim.Vector3r(start_pos_list[0], start_pos_list[1], start_pos_list[2])
                exp_target_position = airsim.Vector3r(target_pos_list[0], target_pos_list[1], target_pos_list[2])

                agent = UAVSearchAgent(
                    start_position=exp_start_position,
                    target_position=exp_target_position,
                    object_name=task['object_name'],
                    object_description=task['description'],
                    log_dir=BASE_LOG_DIR
                )
                agent.run()

                print(f"TASK {task_id} COMPLETED.")
                time.sleep(5)

        except Exception as e:
            print(f"An unexpected error occurred during experiments: {e}")
        finally:
            print("All experiments finished. Cleaning up...")
            self.cleanup()

    def cleanup(self):
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

if __name__ == "__main__":
    MAP_SCRIPTS_CONFIG = {
        "BrushifyUrban": "BrushifyUrban/BrushifyUrban.sh",
        "CabinLake": "CabinLake/CabinLake.sh",
        "DownTown": "DownTown/DownTown_test1.sh",
        "Neighborhood": "Neighborhood/NewNeighborhood.sh",
        "Slum": "Slum/Slum_test1.sh",
        "UrbanJapan": "UrbanJapan/UrbanJapan.sh",
        "Venice": "Venice/Vinice_test1.sh",
        "WesternTown": "WesternTown/WesternTown_test1.sh",
        "WinterTown": "WinterTown/WinterTown_test1.sh",
        "Barnyard": "Barnyard/Barnyard_test1.sh",
        "CityStreet": "CityStreet/CleanCityStreet.sh",
        "NYC": "NYC/NYC1950.sh"
    }

    TASKS_JSON_PATH = "uav_search/task_map/val_tasks.json"
    
    BASE_LOG_DIR = "all4_experiment_logs"

    if not os.path.exists(TASKS_JSON_PATH):
        print(f"Error: Tasks file not found at {TASKS_JSON_PATH}")
    else:
        runner = ExperimentRunner(
            tasks_file=TASKS_JSON_PATH,
            map_scripts_config=MAP_SCRIPTS_CONFIG,
            base_log_dir=BASE_LOG_DIR
        )
        runner.run_all_experiments()

