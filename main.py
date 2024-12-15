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

def generate_challenging_path():
    # Hard-coded path with abrupt changes and occlusions
    path = np.array([
        [0, 0],
        [1, 1],
        [2, 0],
        [3, 3],  # Sudden jump
        [4, 1],
        [5, 5],  # Another jump
        [6, 2],
        [7, 7],  # Large jump
        [8, 3],
        [9, 9],  # Final jump
    ], dtype=np.float64)  # Ensure path is of type float64
    
    # Add non-Gaussian noise
    noise = np.random.laplace(0, 0.5, path.shape)  # Laplace noise
    path += noise
    
    return path

def generate_bad_path(steps):
    """
    Generates a path with non-linear, oscillatory, and sudden movements.
    """
    t = np.linspace(0, 10, steps)
    
    # Non-linear path with oscillations and sudden jumps
    x = 10 * np.sin(t) + np.piecewise(t, [t < 5, t >= 5], [0, lambda t: 20 * np.sin(2 * t)])
    y = 5 * np.cos(t) + np.where(t > 7, 50, 0)  # Sudden jump at t > 7
    
    return np.stack((x, y), axis=1)

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

def add_obstacles_and_return_ids(path):
    obstacle_ids = []
    # Assuming add_obstacles_along_path returns the IDs of the obstacles it creates
    obstacle_ids.extend(add_obstacles_along_path(path))
    return obstacle_ids

def clear_obstacles(obstacle_ids):
    for obstacle_id in obstacle_ids:
        p.removeBody(obstacle_id)

def generate_unpredictable_path(length, num_segments):
    path = []
    current_position = np.array([0, 0])
    
    for _ in range(num_segments):
        # Introduce extreme changes in direction and speed
        segment_length = np.random.uniform(0.1, 2.0)
        angle = np.random.uniform(-2 * np.pi, 2 * np.pi)  # Extreme direction changes
        direction = np.array([np.cos(angle), np.sin(angle)])
        
        # Add non-linear dynamics by varying the segment length randomly
        next_position = current_position + direction * segment_length
        path.append(next_position)
        current_position = next_position
    
    path = np.array(path)
    
    # Add random noise to the path
    noise = np.random.normal(0, 1.0, path.shape)  # High noise level
    path += noise
    
    return path

def generate_extremely_challenging_path(length, num_segments):
    path = []
    current_position = np.array([0, 0])
    
    for _ in range(num_segments):
        # Introduce extreme changes in direction and speed
        segment_length = np.random.uniform(0.1, 2.0)
        angle = np.random.uniform(-2 * np.pi, 2 * np.pi)  # Extreme direction changes
        direction = np.array([np.cos(angle), np.sin(angle)])
        
        # Add non-linear dynamics by varying the segment length randomly
        next_position = current_position + direction * segment_length
        path.append(next_position)
        current_position = next_position
    
    path = np.array(path)
    
    # Add random noise to the path
    noise = np.random.normal(0, 1.0, path.shape)  # High noise level
    path += noise
    
    return path

if __name__ == "__main__":
    initial_state = [0, 0, 0.5]
    
    # Set noise settings for the Kalman filter
    kalman_process_noise = [1000.0, 1000.0, 1000.0]
    measurement_noise = [2000.0, 1000.0, 2000.0]
    
    particle_process_noise = [0.1, 0.1, 0.1]
    num_particles = 500 
    
    # Initialize filters
    kalman_filter = KalmanFilter(initial_state, kalman_process_noise, measurement_noise)
    particle_filter = ParticleFilter(initial_state, num_particles=num_particles, process_noise=particle_process_noise, measurement_noise=measurement_noise)
    
    # Setup simulation
    robot_id = setup_simulation()
    
    # Generate an extremely challenging path
    challenging_path = generate_extremely_challenging_path(length=15, num_segments=10)
    
    # Run simulation for challenging path
    print("Running simulation for extremely challenging path...")
    simulate(robot_id, kalman_filter, particle_filter, challenging_path)

    
