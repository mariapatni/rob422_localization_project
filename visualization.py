import matplotlib.pyplot as plt
import numpy as np

def visualize_particles(particles, weights, true_position=None):
    plt.figure(figsize=(10, 10))
    
    # Normalize weights for better visualization
    normalized_weights = weights / np.max(weights)
    
    # Plot particles
    plt.scatter(particles[:, 0], particles[:, 1], s=normalized_weights * 100, c='blue', alpha=0.5, label='Particles')
    
    # Plot true position if available
    if true_position is not None:
        plt.scatter(true_position[0], true_position[1], c='red', marker='x', s=100, label='True Position')
    
    plt.title('Particle Filter Visualization')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_paths(true_path, kalman_path, particle_path, particles, kalman_error, particle_error):
    
    # PART 1: Simple Path, Kalman Filter Path vs Particle Filter Path
    
    # Create a figure with two subplots, one below the other, with smaller size
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))  # Adjusted size to be smaller
    
    # Plot Kalman Filter Path vs True Path
    axs[0].plot(true_path[:, 0], true_path[:, 1], label='True Path', color='black')
    axs[0].plot(kalman_path[:, 0], kalman_path[:, 1], label='Kalman Filter Path', color='blue')
    axs[0].set_xlabel('X Position')
    axs[0].set_ylabel('Y Position')
    axs[0].set_title('Kalman Filter Path vs True Path')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].text(0.5, -0.15, f"Mean Error: {kalman_error:.4f}", 
                horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes)
    
    # Plot Particle Filter Path vs True Path
    axs[1].plot(true_path[:, 0], true_path[:, 1], label='True Path', color='black')
    axs[1].plot(particle_path[:, 0], particle_path[:, 1], label='Particle Filter Path', color='red')
    axs[1].set_xlabel('X Position')
    axs[1].set_ylabel('Y Position')
    axs[1].set_title('Particle Filter Path vs True Path')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].text(0.5, -0.15, f"Mean Error: {particle_error:.4f}", 
                horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

    # PART 2: Complex Path, Kalman Filter Path vs Particle Filter Path