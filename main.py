import pybullet as p
import pybullet_data
import numpy as np
from kalman_filter import KalmanFilter
from particle_filter import ParticleFilter
from utils import setup_simulation, get_noisy_position, add_obstacles_along_path
from visualization import visualize_paths
import time

def generate_jagged_path(length, num_segments):
    path = []
    current_position = np.array([0, 0])
    direction = 1
    
    for _ in range(num_segments):
        # Random segment length between 0.5 and 1.5
        segment_length = np.random.uniform(0.5, 1.5)
        next_position = current_position + np.array([segment_length, direction * segment_length])
        path.append(next_position)
        current_position = next_position
        direction *= -1  # Alternate direction to create a zigzag pattern
    
    return np.array(path)

def calculate_error(true_path, estimated_path):
    # Calculate the Euclidean distance between corresponding points
    errors = np.linalg.norm(true_path - estimated_path, axis=1)
    mean_error = np.mean(errors)
    return mean_error

def simulate(robot_id, kalman_filter, particle_filter, true_path):
    kalman_path = []
    particle_path = []
    
    # Set the camera view from above with the origin at the bottom of the screen
    p.resetDebugVisualizerCamera(cameraDistance=20, cameraYaw=90, cameraPitch=-90, cameraTargetPosition=[10, 0, 0])
    
    # Add obstacles along the true path
    add_obstacles_along_path(true_path)
    
    # Run the simulation for the length of the true path
    for step in range(len(true_path)):
        # Get true position from the predefined path
        true_position = true_path[step]
        
        # Calculate control input to move towards the next waypoint
        current_position, _ = p.getBasePositionAndOrientation(robot_id)
        control_input = np.array(true_position) - np.array(current_position[:2])
        control_input = np.append(control_input, 0)  # Add zero for the z-axis
        
        # Apply control input to the robot
        p.resetBasePositionAndOrientation(robot_id, current_position + control_input, [0, 0, 0, 1])
        p.stepSimulation()
        
        # Add a delay between simulation steps
        time.sleep(0.1)  # 0.1 seconds delay
        
        # Get noisy position
        noisy_position = get_noisy_position(robot_id)
        
        # Update filters with the same frequency as control inputs and sensor measurements
        # Kalman filter update
        kalman_filter.predict(control_input)
        kalman_filter.update(noisy_position)
        kalman_path.append(kalman_filter.state[:2])
        
        # Particle filter update
        particle_filter.predict(control_input)
        particle_filter.update(noisy_position)
        particle_filter.resample()
        particle_estimate = np.mean(particle_filter.particles, axis=0)
        particle_path.append(particle_estimate[:2])
        
    # Convert paths to numpy arrays for error calculation
    kalman_path = np.array(kalman_path)
    particle_path = np.array(particle_path)
    
    # Calculate and display the errors
    kalman_error = calculate_error(true_path, kalman_path)
    particle_error = calculate_error(true_path, particle_path)
    print(f"Mean Error for Kalman Filter Path: {kalman_error:.4f}")
    print(f"Mean Error for Particle Filter Path: {particle_error:.4f}")
    
    # Visualize paths and final positions
    visualize_paths(np.array(true_path), kalman_path, particle_path, particle_filter.particles, kalman_error, particle_error)
    
    # Compare final estimates
    print("Final Kalman Estimate:", kalman_filter.state)
    print("Final Particle Estimate:", particle_estimate)

if __name__ == "__main__":
    
    initial_state = [0, 0, 0.5]
    
    # Further increase the process and measurement noise for even more deviation
    process_noise = [0.2, 0.2, 0.2]  # Increased from [0.1, 0.1, 0.1]
    measurement_noise = [2.0, 2.0, 2.0]  # Increased from [1.0, 1.0, 1.0]
    
    kalman_filter = KalmanFilter(initial_state, process_noise, measurement_noise)
    
    # Adjust the instantiation of ParticleFilter to match its constructor
    particle_filter = ParticleFilter(initial_state, num_particles=100, process_noise=process_noise, measurement_noise=measurement_noise)
    
    # Generate a jagged true path with random segment lengths
    true_path = generate_jagged_path(length=25, num_segments=25)
    
    robot_id = setup_simulation()
    
    simulate(robot_id, kalman_filter, particle_filter, true_path) 