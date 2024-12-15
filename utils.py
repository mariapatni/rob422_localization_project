import pybullet as p
import pybullet_data
import numpy as np

def setup_simulation():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("models/drake/pr2_description/urdf/pr2_simplified.urdf", basePosition=[0, 0, 0.5])
    
    # Add obstacles to create a corridor for the zigzag path
    obstacle_positions = [
        [1, 2, 0.5], [1, -2, 0.5],
        [3, 2, 0.5], [3, -2, 0.5],
        [5, 2, 0.5], [5, -2, 0.5],
        [7, 2, 0.5], [7, -2, 0.5],
        [9, 2, 0.5], [9, -2, 0.5],
        [2, 0, 0.5], [4, 0, 0.5], [6, 0, 0.5], [8, 0, 0.5]
    ]
    for pos in obstacle_positions:
        p.loadURDF("cube.urdf", basePosition=pos, globalScaling=0.5)
    
    return robot_id

def get_noisy_position(robot_id):
    true_position, _ = p.getBasePositionAndOrientation(robot_id)
    noise = np.random.normal(0, 0.1, size=3)  # Adjust noise level as needed
    noisy_position = np.array(true_position) + noise
    return noisy_position 