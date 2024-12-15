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
    for point in path:
        # Place obstacles on either side of the path
       
        offset = 1.5  # Example offset value
        # Ensure point is 3D by appending a z-coordinate, e.g., 0
        point_3d = np.append(point, 0)
        left_obstacle = point_3d + np.array([0, offset, 0.5])
        right_obstacle = point_3d + np.array([0, -offset, 0.5])
        p.loadURDF("cube.urdf", basePosition=left_obstacle, globalScaling=0.2)
        p.loadURDF("cube.urdf", basePosition=right_obstacle, globalScaling=0.2)

def get_noisy_position(robot_id):
    true_position, _ = p.getBasePositionAndOrientation(robot_id)
    noise = np.random.normal(0, 0.1, size=3)  # Adjust noise level as needed
    noisy_position = np.array(true_position) + noise
    return noisy_position 