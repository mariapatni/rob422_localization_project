import numpy as np

class KalmanFilter:
    def __init__(self, initial_state, process_noise, measurement_noise):
        self.state = np.array(initial_state)
        self.process_noise = np.diag(process_noise)
        self.measurement_noise = np.diag(measurement_noise)
        self.covariance = np.eye(len(initial_state))

    def predict(self, control_input):
        self.state = self.state + control_input
        self.covariance = self.covariance + self.process_noise

    def update(self, measurement):
        kalman_gain = self.covariance @ np.linalg.inv(self.covariance + self.measurement_noise)
        self.state = self.state + kalman_gain @ (measurement - self.state)
        self.covariance = (np.eye(len(self.state)) - kalman_gain) @ self.covariance 