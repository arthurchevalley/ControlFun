import time

import numpy as np
import math

import matplotlib.pyplot as plt


class Kalman:
    """
    Kalman class that calculates the estimation of the position based on odometry and/or measurements.
    """

    def __init__(self, qx=0.1, qy=0.1, qt=np.deg2rad(1.0),Ts=0.1, b=0.01, k_delta_sr=.00001, k_delta_sl=.00001):
        """
        Constructor that initializes the class variables.
        recommended values: qx=2.8948e-04, qy=8.2668e-04, qt=2.9e-03, k_delta_sr=1.3400e-02, k_delta_sl=8.3466e-03

        param qx: variance on the x axis
        param qy: variance on the y axis
        param qt: variance on the angle
        param k_delta_sr: variance on the right speed
        param k_delta_sl: variance on the left speed
        """
        self.Ts = Ts
        self.state_est = [np.array([[0], [0], [0]])]
        self.cov_est = 0.01 * np.ones([3, 3])  # default, do not tune here
        self.b = b
        self.H = np.array([[1, 0, 0], [0, 1, 0]])
        self.stripe_width = 50
        self.qx = qx
        self.qy = qy
        self.qt = qt
        self.k_delta_sr = k_delta_sr
        self.k_delta_sl = k_delta_sl
        self.Q = np.array([[self.qx, 0, 0], [0, self.qy, 0], [0, 0, self.qt]])
        self.R = np.array([[self.k_delta_sr, 0], [0, self.k_delta_sl]])
        
        self.cov_all = []
        self.pos_all = []

    def tune_values(self, qx, qy, qt, k_delta_sr, k_delta_sl):
        """
        Allows to change the variance matrices during the execution

        param qx: variance on the x axis
        param qy: variance on the y axis
        param qt: variance on the angle
        param k_delta_sr: variance on the right speed
        param k_delta_sl: variance on the left speed
        """
        self.qx = qx
        self.qy = qy
        self.qt = qt
        self.k_delta_sr = k_delta_sr
        self.k_delta_sl = k_delta_sl
        self.Q = np.array([[self.qx, 0, 0], [0, self.qy, 0], [0, 0, self.qt]])
        self.R = np.array([[self.k_delta_sr, 0], [0, self.k_delta_sl]])

    def __jacobianf_x(self, theta, delta_s, delta_theta):
        """
        Compute the partial derivative of the motion model with respect to the state vector x, evaluated at the current state x and input u 

        :param theta: current orientation of the robot 
        :param delta_s: mean of the travelled distance of the right wheel and the left wheel 
        :param delta_theta: angle increment based on the travelled distance of the right wheel and the left wheel, and the distance between the wheels

        :return: a matrix (np.array) containing the partial derivative evaluated at the current state and input u 
        """
        return np.array(
            [[1, 0, -delta_s * np.sin(theta + delta_theta / 2)],
             [0, 1, delta_s * np.cos(theta + delta_theta / 2)],
             [0, 0, 1]])

    def __jacobianf_u(self, theta, delta_s, delta_theta):
        """
        Compute the partial derivative of the motion model with respect to the input vector u, evaluated at the current state x and input u 

        :param theta: current orientation of the robot 
        :param delta_s: mean of the travelled distance of the right wheel and the left wheel 
        :param delta_theta: angle increment based on the travelled distance of the right wheel and the left wheel, and the distance between the wheels

        :return: a matrix (np.array) containing the partial derivative evaluated at the current state x and input u 
        """
        return np.array(
            [[1 / 2 * np.cos(theta + delta_theta / 2) - delta_s / (2 * self.b) * np.sin(theta + delta_theta / 2),
              1 / 2 * np.cos(theta + delta_theta / 2) + delta_s / (2 * self.b) * np.sin(theta + delta_theta / 2)],
             [1 / 2 * np.sin(theta + delta_theta / 2) + delta_s / (2 * self.b) * np.cos(theta + delta_theta / 2),
              1 / 2 * np.sin(theta + delta_theta / 2) - delta_s / (2 * self.b) * np.cos(theta + delta_theta / 2)],
             [1 / self.b, -1 / self.b]])
        
    def plot(self):
        state_pred = self.pos_all
        cov_pred = self.cov_all

        plt.ion()
        fig, ax = plt.subplots()

        px, py = self.plot_covariance_ellipse(state_pred[0], cov_pred[0] / 1000)
        line_v = ax.axvline(x=state_pred[0][0], color="k")
        line_h = ax.axhline(y=state_pred[0][1], color="k")
        ellips, = ax.plot(px, py, "--r", label="covariance matrix")

        max_l = len(state_pred)
        l = [i for i in range(max_l)]
        l.insert(0, -1)
        for i in l:
            px, py = self.plot_covariance_ellipse(state_pred[i], cov_pred[i] / 1000)

            line_v.set_xdata(state_pred[i][0])
            line_h.set_ydata(state_pred[i][1])

            ellips.set_xdata(px)
            ellips.set_ydata(py)
            ax.relim()
            ax.autoscale_view()

            fig.canvas.draw()

            fig.canvas.flush_events()
            # plt.axis([0, 0.725, 0, 0.8])
            plt.show()

            time.sleep(4)

    def plot_covariance_ellipse(self, state_est, cov_est):

        Pxy = cov_est[0:2, 0:2]
        eigval, eigvec = np.linalg.eig(Pxy)

        if eigval[0] >= eigval[1]:
            bigind = 0
            smallind = 1
        else:
            bigind = 1
            smallind = 0

        t = np.arange(0, 2 * math.pi + 0.1, 0.1)
        a = math.sqrt(eigval[bigind])
        b = math.sqrt(eigval[smallind])
        x = [a * math.cos(it) for it in t]
        y = [b * math.sin(it) for it in t]

        angle = math.atan2(eigvec[bigind, 1], eigvec[bigind, 0])
        R = np.array([[math.cos(angle), math.sin(angle)],
                      [-math.sin(angle), math.cos(angle)]])
        fx = R.dot(np.array([[x, y]]))
        px = np.array(fx[0, :] + state_est[0]).flatten()
        py = np.array(fx[1, :] + state_est[1]).flatten()

        return px, py

    def kalman_filter(self, z, state_est_prev, cov_est_prev, wheel_traveled):
        """
        Estimates the current state using input sensor data and the previous state
        Everything is in meter and seconds here!

        param z: array representing the measurement (x,y) (coming from the GPS sensor)
        param wheel_traveled: travelled distance for the right wheel and left wheel (in meters)
        param state_est_prev: previous state a posteriori estimation
        param cov_est_prev: previous state a posteriori covariance

        return state_est: new a posteriori state estimation
        return cov_est: new a posteriori state covariance
        """
        if np.any(z) is None:  # if no gps, just odometry
            measurement = False
        else:
            measurement = True
            # estimated mean of the state
            z = np.array([[z[0]], [z[1]]])
            
            
        theta = state_est_prev[2]
        delta_s = (wheel_traveled[0] + wheel_traveled[1]) / 2
        delta_theta = math.atan2(wheel_traveled[0] - wheel_traveled[1], 2*self.b)

        Fx = self.__jacobianf_x(theta, delta_s, delta_theta)
        Fu = self.__jacobianf_u(theta, delta_s, delta_theta)
        
        # Prediction step
        state_est_prev = np.array([[state_est_prev[0]], [state_est_prev[1]], [state_est_prev[2]]])

        state_est_a_priori = state_est_prev + np.array(
            [[delta_s * np.cos(theta + delta_theta / 2)], [delta_s * np.sin(theta + delta_theta / 2)], [delta_theta]])

        # Estimated covariance of the state
        Ppred = Fx @ cov_est_prev @ Fx.T + Fu @ self.R @ Fu.T + self.Q

        if measurement:  # odometry et measurements
            # Update step
            # innovation / measurement residual
            i = z - self.H @ state_est_a_priori
            S = self.H @ Ppred @ self.H.T + self.R
            
            # Kalman gain (tells how much the predictions should be corrected based on the measurements)
            K = Ppred @ self.H.T @ np.linalg.inv(S)

            # a posteriori estimate
            state_est = state_est_a_priori + np.dot(K, i)
            cov_est = Ppred - K @ self.H @ Ppred 

        else:  # odometry
            state_est = state_est_a_priori
            cov_est = Ppred
            
        return state_est.flatten().tolist(), cov_est

def update_mov(state, wheel_traveled, dt, vmax=1, omega=.1, radius=0.01):
    x_update = vmax * np.cos(state[2]) * dt
    y_update = vmax * np.sin(state[2]) * dt
                
    state[0] += x_update
    state[1] += y_update
    state[2] += omega * dt
    r_update = math.sqrt(x_update**2+y_update**2) + omega*dt*radius
    l_update = math.sqrt(x_update**2+y_update**2) - omega*dt*radius

    wheel_traveled[0] = r_update
    wheel_traveled[1] = l_update

    return state, wheel_traveled


NOISE_VARIANCE = 0.05
if __name__ == '__main__':
    kalman = Kalman()

    state = np.array([0.5,0.4,math.pi/2])
    radius = 0.01

    state_est_prev = np.array([[0.5],[0.4], [math.pi/2]])
    cov_est_prev = kalman.cov_est
    wheel_traveled = np.array([0.,0.])
    show_animation = False
    measurement = True
    z_hist = []
    state_hist = []
    state_est_hist = []
    cov_hist = []
    simulationTime = 16 # seconds
    dt = 0.1
    nbr_GPS_drop = 2 # Number of drop during the simulation
    GPS_drop_length_array = [.1, .2, .5, 1., 2., 5., 10., 15.]  # GPS lost signal in seconds
    Lost_probability =[2./8, 1.75/8, 1.5/8, 1./8, .75/8, .5/8, .375/8, .125/8]
    GPS_drop_length = np.random.choice(GPS_drop_length_array, nbr_GPS_drop, Lost_probability)
    print(GPS_drop_length)
    lost_GPS_time = np.random.randint(0, (simulationTime/dt), nbr_GPS_drop)
    sim_step = int(simulationTime/dt)
    for id in range(sim_step):
        state, wheel_traveled = update_mov(state, wheel_traveled, dt)
        if not np.logical_and((lost_GPS_time).astype(int)<= id,(lost_GPS_time+GPS_drop_length/dt).astype(int)>=id).any():
            z = state[:2] + NOISE_VARIANCE*np.random.randn(2)
        else:
            z = np.array([None, None])
            
        wheel_traveled += + 0.01*NOISE_VARIANCE*np.random.randn(2)
        state_est, cov_est = kalman.kalman_filter(z, state_est_prev, cov_est_prev, wheel_traveled)
        state_est_prev = state_est
        cov_est_prev = cov_est

        z_hist.append([z[0],z[1]])
        state_hist.append([state[0],state[1]])
        state_est_hist.append([state_est[0], state_est[1]])
        cov_hist.append(cov_est)
        
        #print(f'actual state {state} \nmeasured state {z}\n estimated state {state_est_prev}\n')


    for id in range(sim_step):
        if z_hist[id][0] is not None:
            plt.plot(z_hist[id][0], z_hist[id][1], "+g")
        px, py = kalman.plot_covariance_ellipse(state_est_hist[id], cov_hist[id])
        plt.plot(px, py, "--r")
        plt.plot(state_hist[id][0], state_hist[id][1], "xb")
        plt.plot(state_est_hist[id][0], state_est_hist[id][1], ".r")
        
    plt.show()
    plt.pause(35)