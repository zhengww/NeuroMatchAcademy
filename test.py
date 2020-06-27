from control import lqr
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

import numpy as np                  # import numpy
import scipy                        # import scipy
import random                       # import basic random number generator functions
from scipy.linalg import inv        # import matrix inversion
import matplotlib.pyplot as plt     # import matplotlib


class LDS:
    def __init__(self, T, ini_state, noise_var, goal):
        self.T = T
        self.goal = goal
        self.ini_state = ini_state
        self.noise_var = noise_var

    def dynamics(self, D, B):

        s = np.zeros(self.T)  # states initialization
        s[0] = self.ini_state

        noise = np.random.normal(0, self.noise_var, self.T)

        for t in range(self.T - 1):
            ###################################################################
            ## Insert your code here to fill with the state dynamics equation
            ## without any control input
            ## s[t+1] = ?
            ###################################################################
            s[t + 1] = 0  # replace 0 with your answer

        return s

    # to_remove solution
    def dynamics_solution(self, D, B):

        s = np.zeros(self.T)  # states initialization
        s[0] = self.ini_state

        noise = np.random.normal(0, self.noise_var, self.T)

        for t in range(self.T - 1):
            s[t + 1] = D * s[t] + noise[t]

        return s

    def dynamics_openloop(self, D, B, a):

        s = np.zeros(self.T)  # states initialization
        s[0] = self.ini_state

        noise = np.random.normal(0, self.noise_var, self.T)

        for t in range(self.T - 1):
            ###################################################################
            ## Insert your code here to fill with the state dynamics equation
            ## with open-loop control input a[t]
            ## s[t+1] = ?
            ###################################################################
            s[t + 1] = 0  # replace 0 with your answer

        return s

    # to_remove solution
    def dynamics_openloop_solution(self, D, B, a):

        s = np.zeros(self.T)  # states initialization
        s[0] = self.ini_state

        noise = np.random.normal(0, self.noise_var, self.T)

        for t in range(self.T - 1):
            s[t + 1] = D * s[t] + B * a[t] + noise[t]

        return s

    def dynamics_closedloop(self, D, B, L):

        s = np.zeros(self.T)  # states initialization
        s[0] = self.ini_state

        noise = np.random.normal(0, self.noise_var, self.T)
        a = np.zeros(self.T)

        for t in range(self.T - 1):
            ###################################################################
            ## Insert your code here to fill with the state dynamics equation
            ## with closed-loop control input as a function of control gain L.
            ## a[t] = ?
            ## s[t+1] = ?
            ###################################################################
            a[t] = 0  # replace 0 with your answer
            s[t + 1] = 0  # replace 0 with your answer

        return s, a

    # to_remove solution
    def dynamics_closedloop_solution(self, D, B, L):

        s = np.zeros(self.T)  # states initialization
        s[0] = self.ini_state

        noise = np.random.normal(0, self.noise_var, self.T)
        a = np.zeros(self.T - 1)

        for t in range(self.T - 1):
            a[t] = L[t] * s[t]
            s[t + 1] = D * s[t] + B * a[t] + noise[t]

        return s, a


class LQR(LDS):
    def __init__(self, T, ini_state, noise_var, goal):
        super().__init__(T, ini_state, noise_var, goal)

    def control_gain_LQR(self, D, B, rho):
        P = np.zeros(self.T)  # Riccati updates
        P[-1] = 1

        L = np.zeros(self.T - 1)  # control gain

        for t in range(self.T - 1):
            P[self.T - t - 2] = (1 +
                                 P[self.T - t - 1] * D ** 2 -
                                 D * P[self.T - t - 1] * B / (
                                         rho + P[self.T - t - 1] * B) * B ** 2 * P[self.T - t - 1] * D)

            L[self.T - t - 2] = -(1 / (rho + P[self.T - t - 1] * B ** 2) * B * P[self.T - t - 1] * D)

        return L, P

    def calculate_J_state(self, s):
        ###################################################################
        ## Insert your code here to calculate J_state(s).
        ## J_state = ?
        ###################################################################
        J_state = 0  # Replace 0 with your answer
        return J_state

    # to remove solution
    def calculate_J_state_solution(self, s):
        J_state = np.sum((s - self.goal) ** 2)
        return J_state

    def calculate_J_control(self, a):
        ###################################################################
        ## Insert your code here to calculate J_control(s).
        ## J_control = ?
        ###################################################################
        J_control = 0  # Replace 0 with your answer
        return J_control

    # to remove solution
    def calculate_J_control_solution(self, a):
        J_control = np.sum(a ** 2)
        return J_control


def plot_vs_time(s,slabel,color,goal=None):
    plt.plot(s, color, label = slabel)
    if goal is not None:
      plt.plot(goal, 'm', label = 'goal')
    plt.xlabel("time", fontsize =14)
    plt.legend(loc="upper right")

## Play around with rho and see the effect on the state and action.
## For which rho is the cost equal to the optimal cost found in Exercise 1?
## Try increasing the rho to 2. What do you notice?
D = 0.9 * np.eye(1) # state parameter
B = 2  * np.eye(1)    # control parameter
rho = 1
T = 20  # time horizon
ini_state = 2      # initial state
noise_var = 0.1   # process noise
goal = np.zeros(T)

lqr0 = LQR(T, ini_state, noise_var, goal)
L, P = lqr0.control_gain_LQR(D, B, rho)
s_lqr, a_lqr = lqr0.dynamics_closedloop_solution(D, B, L)

# with plt.xkcd():
#     fig = plt.figure(figsize=(4, 4))
#     plot_vs_time(s_lqr,'State evolution','b',goal)
#     plt.title('LQR Control')
#     plt.show()
#     fig = plt.figure(figsize=(4, 4))
#     plot_vs_time(a_lqr,'LQR Action','b')
#     plt.show()
#     fig = plt.figure(figsize=(4, 4))
#     plot_vs_time(L,'Control Gain','b')
#     plt.show()


def dlqr(A, B, Q, R):
    """Solve the discrete time lqr controller.

    x[k+1] = A x[k] + B u[k]

    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    # ref Bertsekas, p.151

    # first, try to solve the ricatti equation
    P = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))

    # compute the LQR gain
    K = np.matrix(scipy.linalg.inv(B.T * P * B + R) * (B.T * P * A))

    eigVals, eigVecs = scipy.linalg.eig(A - B * K)

    return K, P, eigVals



# u = np.dot(-K, (state - target))  #2x1


Q=1.0
R=Q * rho

K1, P1, eigVals = dlqr(D, B, Q, R)

K,S,E=lqr(D,B,Q,R)
# BK=np.dot(B,K)
# AminusBK=np.subtract(A,BK)
# sys1=signal.StateSpace(AminusBK,B,C,D)



t1,y1=signal.step(sys1)
plt.plot(t1,y1)
plt.show()