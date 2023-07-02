import numpy as np
from scipy.linalg import expm
import control
import matplotlib.pyplot as plt



def intergrate(A, t1, t2):
    if (np.linalg.det(A)):
        return (expm(A*t2)-np.eye(A.shape[0])) @ np.linalg.inv(A)
    else:
        print("A is singular")
        
        
def simulation():
    # continuous-time system dynamics of the cart-stick balancer
    A = np.array([[0, 1, 0], [31.33, 0, 0.016], [-31.33, 0, -0.216]])
    B = np.array([[0], [-0.649], [8.649]])
    C = np.array([[10, 0, 0]])
    D = np.array([[0]])
    # Parameters of the nominal experiment
    x0 = np.array([[0.2], [0.3], [-0.5]])
    T = 0.1
    Tmax = T * 20
    tau = 0.01
    taumin = tau
    taumax = tau*10

    s1 = T - tau
    s2 = tau
    Gamma_s_1 = intergrate(A, 0, s1)
    Gamma_s_2 = intergrate(A, 0, s2)


    A_new = expm(A * T) + Gamma_s_1 @ B @ C
    B_new = expm(A * s1) @ Gamma_s_2 @ B

    # Nominal controller
    K = np.array([[-556.1829, -208.3171, -12.9905]])

    # Simulation of the NCS
    sist = control.StateSpace(A, B, C, D)
    sistDT = control.c2d(sist, T)

    GammaT = intergrate(A, 0, T)
    sisDTCL = sistDT
    sisDTCL.A = sistDT.A - GammaT @ B @ K

    Ts = T * np.ones(int(Tmax // T))

    Nsteps = len(Ts)
    Nsims = 5
    Yks = []
    Yks_nocomp = []
    Yks_nocomp_full = []
    taus = taumin + (taumax - taumin) * np.random.rand(Nsims)
    for i in range(Nsims):
        
        tks = [0]
        tau = taus[i]
        
            
        # Compute Gammas
        Gamma_tau = intergrate(A, 0, tau)
        Gamma_delta = intergrate(A, 0, T-tau)
        
    
        uks = [0]
        xks = [x0]
        yks = [(C @ x0).item()]

        uks_nocomp = [0]
        xks_nocomp = [x0]
        yks_nocomp = [(C @ x0).item()]

        uks_nocomp_full = [0]
        xks_nocomp_full = [x0]
        yks_nocomp_full = [(C @ x0).item()]
        
        x_nodelay = [x0]
        y_nodelay = [(C @ x_nodelay).item()]

        for j in range(Nsteps - 1):
            tk = tks[-1]
            T = Ts[-1]
            tukm1 = uks[-1]
            xk = xks[-1]
            
            bx_tkptauk = expm(A * tau) @ xk + Gamma_tau @ B * tukm1
            
            # control law with compensation
            tuk = -K @ bx_tkptauk
            
            # control law without compensation
            tuk_nocomp = -K @ xk
            
            # Compute state
            xtkp1 = expm(A * (T - tau)) @ bx_tkptauk + Gamma_delta @ B @ tuk
            xtkp1_nocomp = expm(A * (T - tau)) @ bx_tkptauk + Gamma_delta @ B @ tuk_nocomp
            
            
            # control law without compensation troughouth
            tuk_nocomp_full = -K @ xks_nocomp_full[-1]
            bx_tkptauk_nocomp_full = expm(A * tau) @ xks_nocomp_full[-1] + Gamma_tau @ B * tukm1
            # Compute state
            xtkp1_nocomp_full = expm(A * (T - tau)) @ bx_tkptauk_nocomp_full + Gamma_delta @ B @ tuk_nocomp_full
            
            uks.append(tuk)        
            xks.append(xtkp1)
            yks.append((C @ xtkp1).item())
            
            uks_nocomp.append(tuk_nocomp)
            xks_nocomp.append(xtkp1_nocomp)
            yks_nocomp.append((C @ xtkp1_nocomp).item())
            
            uks_nocomp_full.append(tuk_nocomp_full)
            xks_nocomp_full.append(xtkp1_nocomp_full)
            yks_nocomp_full.append((C @ xtkp1_nocomp_full).item())
            
            
            if i == (Nsims - 1):
                xtk_nodelay = sisDTCL.A @ x_nodelay[-1]
                x_nodelay.append(xtk_nodelay)
                y_nodelay.append((sisDTCL.C @ xtk_nodelay).item())
        
            tks.append(tks[-1]+T)
    
        Yks.append(yks)
        Yks_nocomp.append(yks_nocomp)
        Yks_nocomp_full.append(yks_nocomp_full)
        # Simulation of the NCS in perfect conditions

    return Nsims, tks, y_nodelay, Yks, Yks_nocomp, Yks_nocomp_full

def plotSimulation(Nsims, tks, y_nodelay, Yks, Yks_nocomp, Yks_nocomp_full):
    fig, ax = plt.subplots(3,1)
    ax[0].set_title('Compensation')
    ax[0].step(tks, y_nodelay, linestyle='--', color='blue', linewidth=1, where='post')
    
        
    for i in range(Nsims):
        ax[0].step(tks, Yks[i], linestyle='--', color='green', linewidth=1, where='post')

    ax[0].legend(['No delay','Delay and compensation'])


    ax[1].set_title('No Compensation')
    ax[1].step(tks, y_nodelay, linestyle='--', color='blue', linewidth=1, where='post')

    for i in range(Nsims):
        ax[1].step(tks, Yks_nocomp[i], linestyle='--', color='cyan', linewidth=1, where='post')
    ax[1].legend(['No delay','Delay but no compensation'])

    ax[2].set_title('No Compensation troughout')
    ax[2].step(tks, y_nodelay, linestyle='--', color='blue', linewidth=1, where='post')
    
    for i in range(Nsims):
        ax[2].step(tks, Yks_nocomp_full[i], linestyle='--', color='cyan', linewidth=1, where='post')
    ax[2].legend(['No delay','Delay but no compensation for all simulation'])

    plt.show()

if __name__ == '__main__':
    Nsims, tks, y_nodelay, Yks, Yks_nocomp, Yks_nocomp_full= simulation()
    plotSimulation(Nsims, tks, y_nodelay, Yks, Yks_nocomp, Yks_nocomp_full)