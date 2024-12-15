import pybullet as p
import pybullet_data
import numpy as np

def setup_simulation():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Load a plane to serve as the ground
    plane_id = p.loadURDF("plane.urdf")
    
    # Change the plane's color to green
    p.changeVisualShape(plane_id, -1, rgbaColor=[0, 1, 0, 1])  # RGBA for green
    
    robot_id = p.loadURDF("models/drake/pr2_description/urdf/pr2_simplified.urdf", basePosition=[0, 0, 0.5])
    
    return robot_id

def add_obstacles_along_path(path, offset=0.5):
    obstacle_ids = []  # Initialize a list to store obstacle IDs
    for point in path:
        offset = 1.5  # Example offset value
        point_3d = np.append(point, 0)
        left_obstacle = point_3d + np.array([0, offset, 0.5])
        right_obstacle = point_3d + np.array([0, -offset, 0.5])
        
        # Load obstacles and store their IDs
        left_id = p.loadURDF("cube.urdf", basePosition=left_obstacle, globalScaling=0.2)
        right_id = p.loadURDF("cube.urdf", basePosition=right_obstacle, globalScaling=0.2)
        
        # Append the IDs to the list
        obstacle_ids.extend([left_id, right_id])
    
    return obstacle_ids  # Return the list of obstacle IDs

def clear_obstacles(obstacle_ids):
    for obstacle_id in obstacle_ids:
        p.removeBody(obstacle_id)

def get_noisy_position(robot_id):
    true_position, _ = p.getBasePositionAndOrientation(robot_id)
    noise = np.random.normal(0, 0.1, size=3)  # Adjust noise level as needed
    noisy_position = np.array(true_position) + noise
    return noisy_position 

def clear_all_bodies():
    body_ids = p.getBodyUniqueIdList()
    for body_id in body_ids:
        p.removeBody(body_id)