import numpy as np

from numpy import matlib as mb
import matplotlib.pyplot as plt
import math
import random
from bitstring import BitArray
from shapely.geometry.polygon import Polygon

from shapely.geometry import MultiPoint, Point

# max is either (bias, tau, gain) or (phase offset, period, gain)
from contoller_rep import *
from dijkstra import *
from kalman import *
from RRT import *
MAX_PARAMS = 3
NBR_SENSORS = 8
MAX_RANGE = 4
LASER_NOISE_VARIANCE = 0.05

NO_ROTATION = 0
CW_ROTATION = 1
CCW_ROTATION = -1
import scipy.io as sio

def load_mat_proper(file, UpSample=.5):

    mat = sio.loadmat(file)  # load mat-file
    mdata = mat['path_to_track']  # variable in mat file
    mdtype = mdata.dtype  # dtypes of structures are "unsized objects"
    # * SciPy reads in structures as structured NumPy arrays of dtype object
    # * The size of the array is the size of the structure array, not the number
    #   elements in any particular field. The shape defaults to 2-dimensional.
    # * For convenience make a dictionary of the data using the names from dtypes
    # * Since the structure has only one element, but is 2-D, index it at [0, 0]
    ndata = {n: mdata[n][0, 0] for n in mdtype.names}
    
    upsampXY = []
    upsampYaw = []
    i = 0
    while i < ndata['coordinates'].shape[0] - 1:
        upsampXY.append(ndata['coordinates'][i].tolist())
        upsampYaw.append(pi_2_pi(ndata['orientation'][i].item()))
        if UpSample:
            for j in range(i+1, ndata['coordinates'].shape[0]):
                if np.linalg.norm(ndata['coordinates'][i] - ndata['coordinates'][j]) >= UpSample:
                    i = j
                    break
                elif j == ndata['coordinates'].shape[0]-1:
                    i = j
                    break
        else:
            i += 1
        
    
    upsampXY.append(ndata['coordinates'][-1].tolist())
    upsampYaw.append(pi_2_pi(ndata['orientation'][-1].item()))

    XY = np.array(upsampXY)
    YAW = np.array(np.expand_dims(upsampYaw, axis=1))
    ref = np.append(XY, YAW, axis=1)
    return ref

def pi_2_pi(angle):
    while(angle > math.pi):
        angle = angle - 2.0 * math.pi

    while(angle < -math.pi):
        angle = angle + 2.0 * math.pi

    return angle

class StateClass:
    def __init__(self, x, u, A, B, C, D, dt, xref=None, max_linear_velocity = 3.0, max_angular_velocity = np.pi) -> None:
        self.x = x
        self.x_dot = np.zeros(self.x.shape)
        self.wheel_speed = np.zeros(2)
        
        self.max_linear_velocity = max_linear_velocity # meters per second
        self.max_angular_velocity = max_angular_velocity # radians per second
 
        self.xref = xref
        self.u = u
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.dt = dt
        
    def state_space_model(self, state_t_minus_1, control_input_t_minus_1):
        """
        Calculates the state at time t given the state at time t-1 and
        the control inputs applied at time t-1
        
        :param: A   The A state transition matrix
            3x3 NumPy Array
        :param: state_t_minus_1     The state at time t-1  
            3x1 NumPy Array given the state is [x,y,yaw angle] ---> 
            [meters, meters, radians]
        :param: B   The B state transition matrix
            3x2 NumPy Array
        :param: control_input_t_minus_1     Optimal control inputs at time t-1  
            2x1 NumPy Array given the control input vector is 
            [linear velocity of the car, angular velocity of the car]
            [meters per second, radians per second]
            
        :return: State estimate at time t
            3x1 NumPy Array given the state is [x,y,yaw angle] --->
            [meters, meters, radians]
        """
        # These next 6 lines of code which place limits on the angular and linear 
        # velocities of the robot car can be removed if you desire.
        
        control_input_t_minus_1[0] = np.clip(control_input_t_minus_1[0],-self.max_linear_velocity, self.max_linear_velocity)
        control_input_t_minus_1[1] = np.clip(control_input_t_minus_1[1],-self.max_angular_velocity, self.max_angular_velocity)
        state_estimate_t = (self.A @ state_t_minus_1) + (self.B @ control_input_t_minus_1) 
                
        return state_estimate_t
        
    def UpdateB(self):
        """
        Calculates and returns the B matrix
        3x2 matix ---> number of states x number of control inputs
    
        Expresses how the state of the system [x,y,yaw] changes
        from t-1 to t due to the control commands (i.e. control inputs).
        
        :param yaw: The yaw angle (rotation angle around the z axis) in radians 
        :param deltat: The change in time from timestep t-1 to t in seconds
        
        :return: B matrix ---> 3x2 NumPy array
        """
        self.B = np.array([[np.cos(self.x[2])*self.dt, 0],
                           [np.sin(self.x[2])*self.dt, 0],
                            [0, self.dt]])

    def setXref(self, new_ref):
        self.xref = new_ref
        
class Vehicle:
    def __init__(self, crds = np.array([.5,0.5]), theta=3/4*np.pi, goal = np.array([2.5,-3, np.pi]),radius=0.5, wheelRadius=.05, dt = 0.1, vmax = 1, omega=np.pi/4, sensors_position=None, world=None, nbr_sensors=NBR_SENSORS, max_sensor_range=MAX_RANGE, controller=None, state=None):
        self.crds = crds
        self.radius = radius
        self.goal = goal
        self.goal_radius = 0.1
        self.theta = theta
        self.wheelRadius = wheelRadius
        self.state = state
        self.state.xref = goal.copy()
        self.dt = dt 
        self.vmax = vmax
        self.omega = omega
        self.world = world
        self.controller = controller
        self.nbr_sensors = nbr_sensors
        self.max_sensor_range = max_sensor_range # in meters
        self.prev_sensor_measure = np.ones(nbr_sensors)*MAX_RANGE
        self.sensor_measure = np.ones(nbr_sensors)*MAX_RANGE
        self.coll_id = np.zeros(nbr_sensors)
        # Obstacle avoidance
        self.theta_change = 0
        self.obstacle_cntr = 0
        self.prev_prev_sensor_measure = np.ones(nbr_sensors)*MAX_RANGE
        
        self.SLAM = []
        if sensors_position is None:
            sense_angles = 2*np.pi/NBR_SENSORS    
            self.sensors_position = np.zeros((NBR_SENSORS,3))
            for sense_id in range(NBR_SENSORS):
                self.sensors_position[sense_id][0] = self.crds[0] + self.radius * np.cos(self.theta + sense_id*sense_angles)
                self.sensors_position[sense_id][1] = self.crds[1] + self.radius * np.sin(self.theta+sense_id*sense_angles)
                self.sensors_position[sense_id][2] = self.theta + sense_id*sense_angles
         
    def forward_Kinematics(self):
        k_mat = np.array([[self.self.wheelRadius/2, self.self.wheelRadius/2],
                          [0,0],
                          [self.wheelRadius/(2*self.radius), -self.wheelRadius/(2*self.radius)]])           
        self.state.x_dot = k_mat @ self.state.wheel_speed
        
        
    def inverse_kinematics(self):
        invk_mat = np.array([[1/self.wheelRadius, 0, self.radius/self.wheelRadius],
                             [1/self.wheelRadius, 0, -self.radius/self.wheelRadius]])
        self.state.wheel_speed = invk_mat @ self.state.x_dot
    
    def control(self):
        
        if np.pi/2 < abs((np.arctan2(self.goal[1]-self.crds[1], self.goal[0]-self.crds[0])-self.theta)) <np.pi:
            if self.theta <= 0:
                self.theta_change -= np.pi
                self.theta += np.pi
            else:
                self.theta_change += np.pi
                self.theta -= np.pi
            self.state.x[2] = self.theta
        if self.obstacle_cntr >= 5:
            print("AVOIDANCE")
            rotation = self.obstacle_avoidance()
        else:
            rotation = NO_ROTATION
            self.controller.update(self.state)
        self.theta = self.state.x[2].copy()
                
        sense_angles = 2*np.pi/NBR_SENSORS     
        x_update = self.crds[0] - self.state.x[0]
        y_update = self.crds[1] - self.state.x[1]

        for sense_id in range(NBR_SENSORS):
            self.sensors_position[sense_id][0] = self.state.x[0] + self.radius * np.cos(self.theta+self.theta_change + sense_id*sense_angles)
            self.sensors_position[sense_id][1] = self.state.x[1] + self.radius * np.sin(self.theta+self.theta_change + sense_id*sense_angles)
            self.sensors_position[sense_id][2] = self.theta+self.theta_change + sense_id*sense_angles
        self.UpdateSenor()
        
             
        collision, still_avoid = self.CheckCollision(1.2*self.radius, rotation)

        if any(collision) and self.obstacle_cntr < 5: #still_avoid or self.coll_id[0].shape[0] >= 1:
            self.state.x[0] = self.crds[0].copy()
            self.state.x[1] = self.crds[1].copy()
            self.obstacle_cntr += 1

            for sense_id in range(NBR_SENSORS):
                self.sensors_position[sense_id][0] = self.state.x[0] + self.radius * np.cos(self.theta+self.theta_change + sense_id*sense_angles)
                self.sensors_position[sense_id][1] = self.state.x[1] + self.radius * np.sin(self.theta+self.theta_change + sense_id*sense_angles)
        elif self.obstacle_cntr >= 5 and still_avoid:
            self.crds[0] = self.state.x[0]
            self.crds[1] = self.state.x[1]
        else:
            self.obstacle_cntr = 0
            self.crds[0] = self.state.x[0]
            self.crds[1] = self.state.x[1]
            
        self.UpdateSenor()
        self.UpdateMap()
            
    def CW(self):
        print("CW")
        self.state.x[2] = self.theta - self.omega * self.dt
        self.state.x[:2] = self.crds
    def CCW(self):
        print("CCW")
        self.state.x[2] = self.theta + self.omega * self.dt
        self.state.x[:2] = self.crds
    
    def obstacle_avoidance(self):
        if self.goal[0] > self.crds[0]:             # xg > xc
            if self.goal[1] > self.crds[1]:         # yg > yc
                self.CW()
                return CW_ROTATION
            else:                                   # yg < yc
                self.CCW()
                return CCW_ROTATION
            
        else:                                       # xg < xc
            if self.goal[1] > self.crds[1]:         # yg > yc
                self.CCW()
                return CCW_ROTATION
            else:                                   # yg < yc
                self.CW()
                return CW_ROTATION
                
    def UpdateSenor(self):
        self.prev_sensor_measure = self.sensor_measure.copy()
        tmp_sensor_measure = np.ones(NBR_SENSORS)*MAX_RANGE
        for obs_id, obs in enumerate(self.world.obsPolygon):
            for sense_id in range(NBR_SENSORS):
                cx,cy = polyxline(self.sensors_position[sense_id][:2], [np.cos(self.sensors_position[sense_id][2] + np.pi / 2), np.sin(self.sensors_position[sense_id][2] + np.pi / 2)], self.crds[:2], obs.exterior.xy[0], obs.exterior.xy[1])
                
                dist = MAX_RANGE
                if len(cx)>=1:
                    for id in range(len(cx)):
                        cx[id] += LASER_NOISE_VARIANCE*np.random.randn()
                        cy[id] += LASER_NOISE_VARIANCE*np.random.randn()
                        dist = min(np.linalg.norm([cx[id],cy[id]]-self.sensors_position[sense_id][:2]), dist)
                tmp_sensor_measure[sense_id] = min(min(dist, MAX_RANGE), tmp_sensor_measure[sense_id])
        tmp_sensor_measure[tmp_sensor_measure==MAX_RANGE] = 0
        self.sensor_measure = tmp_sensor_measure.copy()
        
    def UpdateMap(self):
        sense_angles = 2*np.pi/NBR_SENSORS  
        for sense_id in range(NBR_SENSORS):
            if self.sensor_measure[sense_id]:
                sx = self.sensors_position[sense_id][0] + self.sensor_measure[sense_id] * np.cos(self.sensors_position[sense_id][2])
                sy = self.sensors_position[sense_id][1] + self.sensor_measure[sense_id] * np.sin(self.sensors_position[sense_id][2])
                self.SLAM.append([sx,sy])
        
    def CheckCollision(self, threshold, rotation):
        #print(self.prev_sensor_measure - self.sensor_measure)            
        
        dif_mask = ((self.prev_sensor_measure - self.sensor_measure)>0).astype(int)
        dif_mask[2] = 0
        dif_mask[6] = 0

        dif_mask2 = ((self.prev_sensor_measure - self.sensor_measure)<0).astype(int)
        still_avoid= True
        if dif_mask2[2] == 1 and rotation == CW_ROTATION:
            still_avoid = False
        
        elif dif_mask2[6] == 1 and rotation == CCW_ROTATION:
            still_avoid = False
        
        collision = np.logical_and(0 < self.sensor_measure*dif_mask, self.sensor_measure*dif_mask < threshold)
        #if self.obstacle_cntr >= 5:
        #    still_avoid = any(np.logical_and(0 < self.sensor_measure[self.coll_id], self.sensor_measure[self.coll_id] < threshold))
        #else:
        #    still_avoid = False
        #    self.coll_id = np.where(collision.astype(int)==1)

        return collision, still_avoid

    def reset(self, crds = np.array([.5,0.5]), theta=3/4*np.pi,  goal = np.array([2.5,-3, np.pi]),radius=0.5, dt = 0.1, vmax = 1, omega=np.pi/4, sensors_position=None, world=None):
        self.crds = crds
        self.radius = radius
        self.goal = goal
        self.goal_radius = 0.01
        self.theta = theta
        self.dt = dt 
        self.vmax = vmax
        self.omega = omega
        self.world = world
        self.sensor_measure = np.ones(NBR_SENSORS)*MAX_RANGE
        if sensors_position is None:
            sense_angles = 2*np.pi/NBR_SENSORS    
            self.sensors_position = np.zeros((NBR_SENSORS,3))
            for sense_id in range(NBR_SENSORS):
                self.sensors_position[sense_id][0] = self.crds[0] + self.radius * np.cos(self.theta + sense_id*sense_angles)
                self.sensors_position[sense_id][1] = self.crds[1] + self.radius * np.sin(self.theta+sense_id*sense_angles)
                self.sensors_position[sense_id][2] = self.theta + sense_id*sense_angles

    def plot(self,ax, wp=None, show_sense = True, debug=False):
        ax.add_artist(plt.Circle((self.crds[0], self.crds[1]),
                                 self.radius, color='blue', fill=True))
        
        if wp is not None:
            for wp_id in wp:
                ax.add_artist(plt.Circle((wp_id[0], wp_id[1]),
                                 self.goal_radius, color='green', fill=True))
        else:
            ax.add_artist(plt.Circle((self.goal[0], self.goal[1]),
                                 self.goal_radius, color='green', fill=True))
                
        r_tmp = 0.8 * self.radius
        xc = self.crds[0] - np.sqrt(5) * r_tmp / 2 * np.cos(np.arctan(0.5) + self.theta)
        yc = self.crds[1] - np.sqrt(5) * r_tmp / 2 * np.sin(np.arctan(0.5) + self.theta)
        ax.add_artist(plt.Rectangle([xc, yc], 2 * r_tmp, r_tmp, angle=self.theta * 180 / np.pi, facecolor='lime'))
        for sense_id in range(NBR_SENSORS):
            if show_sense:
                if 0 < self.sensor_measure[sense_id]<1:
                    color_sense = 'red'
                elif self.sensor_measure[sense_id] < 2:
                    color_sense = 'orange'
                elif self.sensor_measure[sense_id] < 3:
                    color_sense = 'yellow'
                elif self.sensor_measure[sense_id] < 4:
                    color_sense = 'green'
                else:
                    color_sense = 'blue'
                ax.plot([self.sensors_position[sense_id][0], self.sensors_position[sense_id][0] + (self.sensor_measure[sense_id]) * np.cos(self.sensors_position[sense_id][2])], [self.sensors_position[sense_id][1], self.sensors_position[sense_id][1] + (self.sensor_measure[sense_id]) * np.sin(self.sensors_position[sense_id][2])], color=color_sense)
            ax.add_artist(plt.Circle((self.sensors_position[sense_id][0], self.sensors_position[sense_id][1]), self.radius/10, color='red', fill=True))
            if debug: ax.text(self.sensors_position[sense_id][0], self.sensors_position[sense_id][1], sense_id, fontsize=15)
            
            if len(self.SLAM)>=1:
                for sense_point in self.SLAM:
                    ax.add_artist(plt.Circle((sense_point[0], sense_point[1]), self.radius/10, color='purple', fill=True))

        return ax
 
    def update_goal(self, new_goal):
        self.goal = new_goal
        self.xref = new_goal
        self.state.setXref(self.xref)
        
class World:
    def __init__(self, world_border=2*np.array([[-5,5],[5, 5],[-5,-5],[5,-5]]),obstacles=None):
        self.world_border = world_border
        self.obstacles = obstacles
        self.obsPolygon= []
        if obstacles is not None:
            for obs in obstacles:
                if obs.shape[0] > 3:
                    p = list(Point(obs[0], obs[1]).buffer(obs[2]).exterior.coords)
                else: # Rectangle cx, cy, w, h, theta
                    ux = obs[0] + obs[2]/2
                    uy = obs[1] + obs[3]/2
                    lx = obs[0] - obs[2]/2
                    ly = obs[1] - obs[3]/2
                    p = [[ux,uy],[ux, ly],[lx, uy],[lx,ly]]

                self.obsPolygon.append(Polygon(p))
    def add_obstacles(self, obstacles):
        if self.obstacles is None:
            self.obstacles = obstacles
        else:
            self.obstacles += obstacles  
        for obs in obstacles:
            if obs.shape[0] == 3:
                p = list(Point(obs[0], obs[1]).buffer(obs[2]).exterior.coords)
                
            else: # Rectangle cx, cy, w, h, theta
                ux = obs[0] + obs[2]/2
                uy = obs[1] + obs[3]/2
                lx = obs[0] - obs[2]/2
                ly = obs[1] - obs[3]/2
                p = [[lx,ly],[ux, ly], [ux,uy],[lx, uy]]
            self.obsPolygon.append(Polygon(p))
 
    def plot(self, ax):
        
        ax.set_xlim([self.world_border[:,0].min(), self.world_border[:,0].max()])
        ax.set_ylim([self.world_border[:,1].min(), self.world_border[:,1].max()])

        #ax.add_artist(plt.Rectangle([self.world_border[:,0].min(), self.world_border[:,1].min()], abs(self.world_border[:,0].min())+ abs(self.world_border[:,0].max()), abs(self.world_border[:,1].min())+ abs(self.world_border[:,1].max()),  fill=False))
        for id, obs in enumerate(self.obstacles):
            if obs.shape[0] == 3: # circle cx,cy and radius
                ax.add_artist(plt.Circle((obs[0], obs[1]), obs[2], color='grey', fill=True))

            elif obs.shape[0] == 5: # Rectangle cx, cy, w, h, theta
                xa = (obs[0]) - obs[2]/2
                ya = (obs[1] )- obs[3]/2
                ax.add_artist(plt.Rectangle([xa, ya], obs[2], obs[3], facecolor='grey'))
                ax.text(obs[0], obs[1], id, fontsize=15)
            x,y = self.obsPolygon[id].exterior.xy
            ax.plot(x,y, color='purple')
        return ax
    
def polydist(x, y, v_x, v_y):
    # Calculate distance of points in x, y to polygon with vertices x_v, y_v
    dx = np.roll(v_x, 1) - v_x
    dy = np.roll(v_y, 1) - v_y
    if not isinstance(x, np.ndarray) and not isinstance(x, list):
        x = np.array([x])
        y = np.array([y])
    n = len(x)
    v = len(v_x)
    m_dx = mb.repmat(dx, n, 1)
    m_dy = mb.repmat(dy, n, 1)
    m_v_x = mb.repmat(v_x, n, 1)
    m_v_y = mb.repmat(v_y, n, 1)
    m_x = mb.repmat(x, v, 1).transpose()
    m_y = mb.repmat(y, v, 1).transpose()

    # Compute the convex combination coefficients that define the closest
    # points on polygon rims and input points
    with np.errstate(divide='ignore', invalid='ignore'):  # Zero division is handled in the next line
        A = np.divide(np.multiply(m_dx, m_x - m_v_x) + np.multiply(m_dy, m_y - m_v_y), m_dx ** 2 + m_dy ** 2)
    A = np.maximum(np.minimum(np.nan_to_num(A, nan=np.inf), np.ones(A.shape)), np.zeros(A.shape))

    # Minimum distance to polygon rims
    m_d = (np.multiply(A, m_dx) + m_v_x - m_x) ** 2 + (np.multiply(A, m_dy) + m_v_y - m_y) ** 2
    dist = np.sqrt(np.min(m_d, axis=1))
    ind = np.argmin(m_d, axis=1)

    # Closest points on the polygon boundary
    c_x = np.empty(n)
    c_y = np.empty(n)
    for i, ind_row in enumerate(ind):
        c_x[i] = A[i, ind_row] * m_dx[i, ind_row] + m_v_x[i, ind_row]
        c_y[i] = A[i, ind_row] * m_dy[i, ind_row] + m_v_y[i, ind_row]

    # Change signs for points inside polygon [v_x, v_y]
    poly = MultiPoint([(v_x[i], v_y[i]) for i in range(v)]).convex_hull
    for i in range(n):
        if Point(x[i], y[i]).within(poly):
            dist[i] = -dist[i]
    return c_x.squeeze(), c_y.squeeze(), dist.squeeze()

def polyxline(m, n, o, v_x, v_y):
    # Calculate intersection of line through m with normal n and greater than o with polygon defined by vertices x_v, y_v 

    assert len(v_x) == len(v_y)
    c_x = []
    c_y = []
    for k in range(len(v_x)):
        s_k = ((v_x[k] - m[0]) * n[0] + (v_y[k] - m[1]) * n[1])
        if s_k == 0:
            if np.linalg.norm([v_x[k], v_y[k]] - m) < np.linalg.norm([v_x[k], v_y[k]] - o):
                c_x.append(v_x[k])
                c_y.append(v_y[k])
        else:
            c_n = (k + 1) % len(v_x)
            s_n = ((v_x[c_n] - m[0]) * n[0] + (v_y[c_n] - m[1]) * n[1])
            if s_k * s_n < 0:
                a = -s_n / (s_k - s_n)
                if (a >= 0) and (a <= 1):
                    c_x_tmp = a * v_x[k] + (1 - a) * v_x[c_n]
                    c_y_tmp = a * v_y[k] + (1 - a) * v_y[c_n]
                    if np.linalg.norm([c_x_tmp,c_y_tmp] - m) < np.linalg.norm([c_x_tmp,c_y_tmp] - o):
                        c_x.append(c_x_tmp)
                        c_y.append(c_y_tmp)
    return np.array(c_x), np.array(c_y)


def test_lqr():
    path = os.getcwd() + '/demos/'
    if not os.path.exists(path):
    # If it doesn't exist, create it
        os.makedirs(path)

    
    world = World()
    
    dt = .1
    simulationLength = 20
    lqr = LQR_Control(R = np.array([[0.01,   0],
                                    [  0, 0.01]]), 
                      Q = np.array([[10.0, 0, 0],  # Penalize X position error 
                                    [0, 10.0, 0],  # Penalize Y position error 
                                    [0, 0, 1.0]]), # Penalize YAW ANGLE heading error )
                      N=100
    )
    x0 = np.array([0.,0., 0])
    u0 = np.array([0.,0.])
    A = np.array([[1.0,  0,   0], [0, 1.0, 0], [0, 0, 1.0]])
    B = np.array([[np.cos(x0[2]), 0],[np.sin(x0[2]), 0], [0, 1]])*dt
    
    
    way_points = [np.array([2,-2, -np.pi/2]), np.array([2,2, -np.pi/2]), np.array([-2,2, -np.pi/2])]
    current_goal = 0
    
    uniState = StateClass(x=x0, u=u0, A=A, B=B, C =np.eye(3),D=0, dt=dt)
    uni = Vehicle(crds=x0[:2], theta=x0[2],  goal=way_points[current_goal], world=world, nbr_sensors=8, state=uniState, controller=lqr)
    
    
    obstacles = [np.array([2,2,2,1,math.pi/4]),
                 np.array([-2,-2,2,2, -math.pi/2]),
                 np.array([3,3,1,1,0]),
                 np.array([2,-1,1,1, -math.pi/2])]
    obstacles = []
    world.add_obstacles(obstacles)
    uni.UpdateSenor()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    print_allow=True
    for i in range(int(simulationLength/dt)):
        uni.control()
        if print_allow:
            ax.clear()
            ax = uni.plot(ax)
            ax = world.plot(ax)
            ax.grid(True)
            plt.pause(1)
        if np.linalg.norm(uni.crds-uni.goal[:2]) < uni.radius:
            current_goal += 1
            if current_goal == len(way_points):
                print("done")
                if print_allow: plt.show()
                break
            else:
                print("next waypoint")
                uni.update_goal(way_points[current_goal])

def test_pid():
    path = os.getcwd() + '/demos/'
    if not os.path.exists(path):
    # If it doesn't exist, create it
        os.makedirs(path)


    world = World()
    
    dt = .1
    simulationLength = 100
    pid_cont = PID(kp_linear = 0.5, kd_linear = 0.1, ki_linear = 0,
                        kp_angular = 3, kd_angular = 0.1, ki_angular = 0)


    x0 = np.array([0.,0., 0])
    u0 = np.array([0.,0.])
    A = np.array([[1.0,  0,   0], [0, 1.0, 0], [0, 0, 1.0]])
    B = np.array([[np.cos(x0[2]), 0],[np.sin(x0[2]), 0], [0, 1]])*dt
    
    
    way_points = [np.array([4,-4, -np.pi/2]), np.array([4,4, -np.pi/2]), np.array([-4,4, -np.pi/2])]
    way_points = load_mat_proper('path_2.mat', UpSample=3)
    current_goal = 0
    
    uniState = StateClass(x=x0, u=u0, A=A, B=B, C =np.eye(3),D=0, dt=dt)
    uni = Vehicle(crds=x0[:2], theta=x0[2],  goal=way_points[current_goal], world=world, nbr_sensors=8, state=uniState, controller=pid_cont)
    
    
    obstacles = [np.array([2,2,2,1,math.pi/4]),
                 np.array([-2,-2,2,2, -math.pi/2])]#,
                 #np.array([3,3,1,1,0]),
                 #np.array([2,-1,1,1, -math.pi/2])]
    obstacles = []
    world.add_obstacles(obstacles)
    uni.UpdateSenor()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    print_allow=True
    for i in range(int(simulationLength/dt)):
        
        uni.control()
        if print_allow:
            ax.clear()
            ax.plot(way_points[:,0], way_points[:,1])
            ax = uni.plot(ax)
            ax = world.plot(ax)
            ax.grid(True)

            plt.pause(0.01)
        if np.linalg.norm(uni.crds-uni.goal[:2]) < uni.radius/2:
            current_goal += 1
            if current_goal == len(way_points):
                print("done")
                if print_allow: plt.show()
                break
            else:
                print("next waypoint")
                uni.update_goal(way_points[current_goal])
       
def test_pid_kalman():
    path = os.getcwd() + '/demos/'
    if not os.path.exists(path):
    # If it doesn't exist, create it
        os.makedirs(path)


    world = World()

    dt = .1
    simulationLength = 20
    
    pid_cont = PID(kp_linear = 0.5, kd_linear = 0.1, ki_linear = 0, kp_angular = 3, kd_angular = 0.1, ki_angular = 0)
    way_points = [np.array([-4,-4, 0]), np.array([4,-4, -np.pi/2]), np.array([4,4, -np.pi/2]), np.array([-4,4, -np.pi/2]),np.array([-4,-4, -np.pi/2])]
    #way_points = load_mat_proper('path_2.mat', UpSample=3)
    current_goal = 0
    goal = np.array([-4,4, -np.pi/2])
    x0 = np.array([-4,-4, .0])
    u0 = np.array([0.,0.])
    A = np.array([[1.0,  0,   0], [0, 1.0, 0], [0, 0, 1.0]])
    B = np.array([[np.cos(x0[2]), 0],[np.sin(x0[2]), 0], [0, 1]])*dt
    

    obstacles = [np.array([2,2,2,1,math.pi/4]),
                 np.array([-2,-2,2,2, -math.pi/2])]#,
                 #np.array([3,3,1,1,0]),
                 #np.array([2,-1,1,1, -math.pi/2])]
    obstacles = []
    
    obstacles_border = [np.array([0,-7,7,4,math.pi/4]),
                        np.array([6,.5,2,11, -math.pi/2]),
                        np.array([-4,0,2,2, -math.pi/2]),
                        np.array([-7,.5,2,11, -math.pi/2]),
                        np.array([0,0,6,6,-math.pi/2])]
                        #np.array([2,-1,1,1, -math.pi/2])]
                        
                        
                            # start and goal position


    # set obstacle positions
    ox, oy = [], []
    for obs in obstacles_border:
        for i in range(int(obs[0]-obs[2]/2), int(obs[0]+obs[2]/2)):
            ox.append(i)
            oy.append(int(obs[1]-obs[3]/2))
            ox.append(i)
            oy.append(int(obs[1]+obs[3]/2))
        for i in range(int(obs[1]-obs[3]/2), int(obs[1]+obs[3]/2)):
            oy.append(i)
            ox.append(int(obs[0]-obs[2]/2))
            oy.append(i)
            ox.append(int(obs[0]+obs[2]/2))

    plt.figure()
    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k")
        plt.plot(x0[0], x0[1], "og")
        plt.plot(goal[0], goal[1], "xb")
        plt.grid(True)
        plt.axis("equal")
    radius = 0.5
    dijkstra = Dijkstra(ox, oy, 2.1*radius, radius)
    rx, ry = dijkstra.planning(x0[0], x0[1], goal[0], goal[1])
    rtheta=[-np.pi/2 for i in range(len(rx))]
    way_points = np.array([ry,rx, rtheta]).T

    print(way_points[0])
    

    
    kalman = Kalman(Ts = dt, b=2*radius)
    
    uniState = StateClass(x=x0.copy(), u=u0, A=A, B=B, C =np.eye(3),D=0, dt=dt)
    uni = Vehicle(crds=x0[:2].copy(), theta=x0[2],  goal=way_points[current_goal], world=world, nbr_sensors=8, state=uniState, controller=pid_cont, radius=radius)
    
    

    world.add_obstacles(obstacles_border)
    uni.UpdateSenor()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    print_allow=False
    
    state_est_prev = x0.copy()
    cov_est_prev = kalman.cov_est
    wheel_traveled = np.array([0.,0.])
    
    
    measurement = True
    z_hist = []
    state_hist = []
    state_est_hist = []
    cov_hist = []


    nbr_GPS_drop = 2 # Number of drop during the simulation
    GPS_drop_length_array = [.1, .2, .5, 1.]  # GPS lost signal in seconds
    Lost_probability =[2./8, 1.75/8, 1.5/8, 1./8]
    GPS_drop_length = np.random.choice(GPS_drop_length_array, nbr_GPS_drop, Lost_probability)

    lost_GPS_time = np.random.randint(0, (simulationLength/dt), nbr_GPS_drop)        
    total_trav=0
    total_center_trav =0
    old_crds = x0[:2].copy()
    

    for i in range(int(simulationLength/dt)):
        
        uni.control()
        xy_update = (uni.state.u[0]) * dt

        r_update = xy_update + uni.state.u[1] *dt*uni.radius
        l_update = xy_update - uni.state.u[1] *dt*uni.radius

        wheel_traveled[0] = r_update
        wheel_traveled[1] = l_update
        
        total_trav += (r_update + l_update)/2
        
        total_center_trav += np.linalg.norm(uni.crds-old_crds)
        old_crds = uni.crds.copy()
        if not np.logical_and((lost_GPS_time).astype(int)<= i,(lost_GPS_time+GPS_drop_length/dt).astype(int)>=i).any():
            z = uni.state.x[:2] + 0.1*NOISE_VARIANCE*np.random.randn(2)
        else:
            z = np.array([None, None])
            
        wheel_traveled +=  0.1*NOISE_VARIANCE*np.random.randn(2)
        state_est, cov_est = kalman.kalman_filter(z, state_est_prev, cov_est_prev, wheel_traveled)
        uni.state.x = np.array(state_est)
        uni.crds = uni.state.x[:2]

        uni.theta = uni.state.x[2]

        
        state_est_prev = state_est
        cov_est_prev = cov_est
        
        z_hist.append([z[0],z[1]])
        state_hist.append([uni.state.x[0],uni.state.x[1]])
        state_est_hist.append([state_est[0], state_est[1]])
        cov_hist.append(cov_est)
        
        #print(f'actual state {state} \nmeasured state {z}\n estimated state {state_est_prev}\n')

        
        if print_allow:
            ax.clear()
            #ax.plot(way_points[:,0], way_points[:,1])
            ax = uni.plot(ax)
            ax = world.plot(ax)
            px, py = kalman.plot_covariance_ellipse([state_est[0], state_est[1]], cov_est)
            ax.plot(px, py, "--r")
            
            ax.plot(state_est[0], state_est[1], ".r")
            ax.grid(True)

            plt.pause(0.01)
        if np.linalg.norm(uni.crds-uni.goal[:2]) < uni.radius*0.6:
            
            current_goal += 1
            if current_goal == len(way_points):
                print("done")
                current_goal = 0
                uni.update_goal(way_points[current_goal])
                if print_allow: plt.show()
                break
            else:
                print("next waypoint")
                uni.update_goal(way_points[current_goal])
    

    
    plt.close()
    plt.figure()
    for wp in way_points:
        plt.plot(wp[0], wp[1], ".g", markersize=12)
    for id in range(i):
        if z_hist[id][0] is not None:
            plt.plot(z_hist[id][0], z_hist[id][1], "+g")
        px, py = kalman.plot_covariance_ellipse(state_est_hist[id], cov_hist[id])
        plt.plot(px, py, "--r")
        plt.plot(state_hist[id][0], state_hist[id][1], "xb")
        plt.plot(state_est_hist[id][0], state_est_hist[id][1], ".r")
    if len(uni.SLAM)>=1:
        for sense_point in uni.SLAM:
            plt.plot(sense_point[0], sense_point[1], 'xg')
    plt.show()

def test_lqr_kalman():
    path = os.getcwd() + '/demos/'
    if not os.path.exists(path):
    # If it doesn't exist, create it
        os.makedirs(path)


    world = World()

    dt = .1
    simulationLength = 20
    
    # Optional Variables

    
    lqr = LQR_Control(R = np.array([[0.01,   0],
                                    [  0, 0.01]]), 
                      Q = np.array([[10.0, 0, 0],  # Penalize X position error 
                                    [0, 10.0, 0],  # Penalize Y position error 
                                    [0, 0, 1.0]]), # Penalize YAW ANGLE heading error )
                      N=100
    )
    way_points = [np.array([-4,-4, 0]), np.array([4,-4, -np.pi/2]), np.array([4,4, -np.pi/2]), np.array([-4,4, -np.pi/2])]
    way_points = [np.array([-4,-4, 0]), np.array([0,-4, -np.pi/2]), np.array([4,-4, -np.pi/2]), np.array([4,0, -np.pi/2]),
                  np.array([4,4, 0]), np.array([0,4, -np.pi/2]), np.array([-4,4, -np.pi/2]), np.array([-4,0, -np.pi/2]), np.array([-4,-4, -np.pi/2])]

    #way_points = load_mat_proper('path_2.mat', UpSample=3)
    current_goal = 1

    x0 = np.array([-4,-4, .0])
    u0 = np.array([0.,0.])
    A = np.array([[1.0,  0,   0], [0, 1.0, 0], [0, 0, 1.0]])
    B = np.array([[np.cos(x0[2]), 0],[np.sin(x0[2]), 0], [0, 1]])*dt
    
    
    radius = 0.5
    kalman = Kalman(Ts = dt, b=2*radius)
    
    uniState = StateClass(x=x0.copy(), u=u0, A=A, B=B, C =np.eye(3),D=0, dt=dt)
    uni = Vehicle(crds=x0[:2].copy(), theta=x0[2],  goal=way_points[current_goal], world=world, nbr_sensors=8, state=uniState, controller=lqr, radius=radius)
    
    
    obstacles = [np.array([2,2,2,1,math.pi/4]),
                 np.array([-2,-2,2,2, -math.pi/2])]#,
                 #np.array([3,3,1,1,0]),
                 #np.array([2,-1,1,1, -math.pi/2])]
    #obstacles = []
    world.add_obstacles(obstacles)
    uni.UpdateSenor()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    print_allow=True
    
    state_est_prev = x0.copy()
    cov_est_prev = kalman.cov_est
    wheel_traveled = np.array([0.,0.])
    
    
    measurement = True
    z_hist = []
    state_hist = []
    state_est_hist = []
    cov_hist = []


    nbr_GPS_drop = 4 # Number of drop during the simulation
    GPS_drop_length_array = [.1, .2, .5, 1.]  # GPS lost signal in seconds
    Lost_probability =[2./8, 1.75/8, 1.5/8, 1./8]
    GPS_drop_length = np.random.choice(GPS_drop_length_array, nbr_GPS_drop, Lost_probability)

    lost_GPS_time = np.random.randint(0, (simulationLength/dt), nbr_GPS_drop)        
    total_trav=0
    total_center_trav =0
    old_crds = x0[:2].copy()
    

    for i in range(int(simulationLength/dt)):
        
        uni.control()
        xy_update = (uni.state.u[0]) * dt

        r_update = xy_update + uni.state.u[1] *dt*uni.radius
        l_update = xy_update - uni.state.u[1] *dt*uni.radius

        wheel_traveled[0] = r_update
        wheel_traveled[1] = l_update
        
        total_trav += (r_update + l_update)/2
        
        total_center_trav += np.linalg.norm(uni.crds-old_crds)
        old_crds = uni.crds.copy()
        if not np.logical_and((lost_GPS_time).astype(int)<= i,(lost_GPS_time+GPS_drop_length/dt).astype(int)>=i).any():
            z = uni.state.x[:2] + 0.1*NOISE_VARIANCE*np.random.randn(2)
        else:
            z = np.array([None, None])
            
        wheel_traveled +=  0.1*NOISE_VARIANCE*np.random.randn(2)
        state_est, cov_est = kalman.kalman_filter(z, state_est_prev, cov_est_prev, wheel_traveled)
        uni.state.x = np.array(state_est)
        uni.crds = uni.state.x[:2]

        uni.theta = uni.state.x[2]

        
        state_est_prev = state_est
        cov_est_prev = cov_est
        
        z_hist.append([z[0],z[1]])
        state_hist.append([uni.state.x[0],uni.state.x[1]])
        state_est_hist.append([state_est[0], state_est[1]])
        cov_hist.append(cov_est)
        
        #print(f'actual state {state} \nmeasured state {z}\n estimated state {state_est_prev}\n')

        
        if print_allow:
            ax.clear()
            #ax.plot(way_points[:,0], way_points[:,1])
            ax = uni.plot(ax, wp=way_points)
            ax = world.plot(ax)
            px, py = kalman.plot_covariance_ellipse([state_est[0], state_est[1]], cov_est)
            ax.plot(px, py, "--r")
            
            ax.plot(state_est[0], state_est[1], ".r")
            ax.grid(True)

            plt.pause(0.01)
        if np.linalg.norm(uni.crds-uni.goal[:2]) < uni.radius*0.6:
            current_goal += 1
            if current_goal == len(way_points):
                print("done")
                current_goal = 0
                uni.update_goal(way_points[current_goal])
                #if print_allow: plt.show()
                break
            else:
                print("next waypoint")
                uni.update_goal(way_points[current_goal])
    
    print(total_trav)
    print(total_center_trav)
    plt.close()
    plt.figure()
    for wp in way_points:
        plt.plot(wp[0], wp[1], ".g", markersize=12)
    for id in range(i):
        if z_hist[id][0] is not None:
            plt.plot(z_hist[id][0], z_hist[id][1], "+g")
        px, py = kalman.plot_covariance_ellipse(state_est_hist[id], cov_hist[id])
        plt.plot(px, py, "--r")
        plt.plot(state_hist[id][0], state_hist[id][1], "xb")
        plt.plot(state_est_hist[id][0], state_est_hist[id][1], ".r")
        
    plt.show()


TOTAL_BIT = 10
SAVE_PERIOD = 10
from tqdm import tqdm 
import os
if __name__ == '__main__':
    np.random.seed(42)
    #test_pid_kalman()
    test_lqr_kalman()
   #test_pid()
   #test_lqr()
