import numpy as np

class ParticleFilter:
    def __init__(self, initial_state, num_particles, process_noise, measurement_noise):
        self.num_particles = num_particles
        self.particles = np.tile(initial_state, (num_particles, 1))
        self.weights = np.ones(num_particles) / num_particles
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def predict(self, control_input):
        # Add control input to each particle with process noise
        noise = np.random.normal(0, self.process_noise, self.particles.shape)
        self.particles += control_input + noise

    def update(self, measurement):
        # Calculate weights based on measurement likelihood
        diff = self.particles - measurement
        likelihood = np.exp(-0.5 * np.sum((diff / self.measurement_noise) ** 2, axis=1))
        self.weights = likelihood / np.sum(likelihood)

    def resample(self):
        # Resample particles based on weights
        indices = np.random.choice(range(self.num_particles), size=self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)

    def estimate(self):
        # Return the weighted mean of the particles
        return np.average(self.particles, weights=self.weights, axis=0)
