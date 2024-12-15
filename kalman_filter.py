import numpy as np

class KalmanFilter:
    def __init__(self, initial_state, process_noise, measurement_noise):
        self.state = np.array(initial_state)
        self.process_noise = np.diag(process_noise) * 2.55
        self.measurement_noise = np.diag(measurement_noise) * 5
        self.covariance = np.eye(len(initial_state)) * 1000

    def predict(self, control_input):
        self.state = self.state + control_input
        self.covariance = self.covariance + self.process_noise * 10

    def update(self, measurement):
        S = self.covariance * 10 + self.measurement_noise * 10
        kalman_gain = (self.covariance * 10) @ np.linalg.inv(S)
        
        self.state = self.state + kalman_gain @ (measurement - self.state)
        
        I = np.eye(len(self.state))
        self.covariance = (I - kalman_gain) @ self.covariance