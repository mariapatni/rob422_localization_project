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

def visualize_paths(true_path, kalman_path, particle_path, particles):
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    
    # Determine the limits for the plots
    all_x = np.concatenate((true_path[:, 0], kalman_path[:, 0], particle_path[:, 0]))
    all_y = np.concatenate((true_path[:, 1], kalman_path[:, 1], particle_path[:, 1]))
    x_min, x_max = all_x.min() - 1, all_x.max() + 1
    y_min, y_max = all_y.min() - 1, all_y.max() + 1
    
    # Kalman Filter Path Visualization
    axs[0].plot(true_path[:, 0], true_path[:, 1], 'g-', label='True Path', linewidth=2)
    axs[0].plot(kalman_path[:, 0], kalman_path[:, 1], color='blue', linestyle='--', linewidth=2, alpha=0.6, label='Kalman Filter Path')
    axs[0].set_xlim(x_min, x_max)
    axs[0].set_ylim(y_min, y_max)
    axs[0].set_xlabel('X Position')
    axs[0].set_ylabel('Y Position')
    axs[0].set_title('Kalman Filter vs True Path')
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    axs[0].grid(True)
    axs[0].set_aspect('equal', adjustable='box')
    
    # Particle Filter Path Visualization
    axs[1].plot(true_path[:, 0], true_path[:, 1], 'g-', label='True Path', linewidth=2)
    axs[1].plot(particle_path[:, 0], particle_path[:, 1], color='red', linestyle='-.', linewidth=2, alpha=0.6, label='Particle Filter Path')
    axs[1].set_xlim(x_min, x_max)
    axs[1].set_ylim(y_min, y_max)
    axs[1].set_xlabel('X Position')
    axs[1].set_ylabel('Y Position')
    axs[1].set_title('Particle Filter vs True Path')
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    axs[1].grid(True)
    axs[1].set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.show() 