import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import quad

# Define your class as-is (with slight modifications to remove inline imports)
class Critical_Insolation():
    def __init__(self, ao=0.4, ai=0.6, phi_c_deg=20, Fs=1360/np.pi, A=-218.4492, B=1.4364, T_f=280, k_i=2, adv_flux=0.05):
        self.ao = ao
        self.ai = ai
        self.phi_c = phi_c_deg/180*np.pi
        self.Fs = Fs
        self.A = A
        self.B = B
        self.T_f = T_f

        # Constants and Parameters
        SHR_CONST_RHOICE  = 0.917e3 # density of ice             ~ kg/m^3
        SHR_CONST_LATICE = 3.337e5  # latent heat of fusion     ~ J/kg

        # Physical Constants
        self.sigma = 5.670374419e-8  # Stefan-Boltzmann constant, W/(m^2·K^4)
        self.alpha_i = ai  # Ice albedo
        self.S0 = Fs       # Solar constant, W/m^2
        self.adv_flux = adv_flux
        self.k_i = k_i     # Sea ice thermal conductivity, W/(m·K)
        self.L_i = SHR_CONST_LATICE*SHR_CONST_RHOICE # Sea ice latent heat of fusion, J/m^3

        # Convert units for zeta if needed (as you had (as_u.J).to(as_u.W*as_u.year) ... )
        # If you don't have these unit conversions defined, comment them out or adjust accordingly.
        # For now, let's assume zeta just needs the given expression:
        # This might need to be adjusted or simplified since 'as_u' is not defined.
        # For demonstration, let's just define zeta = (k_i * L_i / B)
        self.zeta = (self.k_i * self.L_i / self.B)

    def F_s_ice_free(self):
        colbedo = (1-self.ao)*(2*self.phi_c+np.sin(2*self.phi_c))/np.pi
        return (self.A+self.B*self.T_f)*4*np.sin(self.phi_c)/colbedo/1360

    # Functions for orbital mechanics
    def r(self,theta, e, a):
        """Calculate orbital distance r as a function of true anomaly theta and eccentricity e."""
        return a * (1 - e**2) / (1 - e * np.cos(theta))

    def F(self,thetas, e, S, a):
        """Calculate net energy flux F(t) at the equator."""
        r_vals = self.r(thetas, e, a)
        adv = self.adv_flux*e/0.2*S*self.S0
        S_r = (1 - self.alpha_i) * S* self.S0 * (a / r_vals)**2 - adv
        F_t = S_r - self.A - self.B * self.T_f 
        return F_t

    def E_anomaly(self,theta, e):
        """Calculate eccentric anomaly E from true anomaly theta."""
        E = 2 * np.arctan(np.tan(theta / 2) * np.sqrt((1 - e) / (1 + e)))
        E = np.mod(E, 2 * np.pi)  # Ensure E is between 0 and 2π
        return E

    def time_from_theta(self,theta, e, P):
        """Calculate time t from true anomaly theta using Kepler's equation."""
        E = self.E_anomaly(theta, e)
        M = E - e * np.sin(E)  # Mean anomaly
        t = P * M / (2 * np.pi)
        return t

    def n_mean_motion(self,theta, e, a):
        """Calculate angular velocity component for time scaling."""
        mu = 4*np.pi**2    # Gravitational parameter for the sun, Msun^3/AU^2
        a = 1.0  # Semi-major axis in AU
        r_val = self.r(theta, e, a)
        n_val = (mu * a * (1 - e**2))**0.5 / r_val**2
        return n_val

    # Energy balance equation
    def energy_balance(self,S,e=0.1,alpha_i=0.6):
        a = (S*np.sqrt(1-e**2))**(-1/2)
        P = a**3/2

        theta = np.linspace(0, 2 * np.pi, 10000)
        F_t = self.F(theta, e, S, a)
        # Find indices where F(t) crosses zero
        idx_crossings = np.where(np.diff(np.sign(F_t)))[0]

        if len(idx_crossings) < 2:
            return np.inf  # No valid solution

        theta1 = theta[idx_crossings[0]]
        theta2 = 2*np.pi - theta1
        t1 = self.time_from_theta(theta1, e, P)
        t2 = P - t1

        F_r = (1-self.alpha_i)*S*self.S0*P/(2*np.pi*a**2*np.sqrt(1-e**2))
        OLR = self.A + self.B*self.T_f

        if theta1 > np.pi:
            lhs = 2*(F_r*theta1 - OLR*t1)
            rhs = 0
        else:
            E1 = F_r*(theta2-theta1) - OLR*(t2-t1)
            E2 = 2*(F_r*theta1 - OLR*t1) 
            lhs = - E1 - E1**2/(2*self.zeta)
            rhs = E2 
        return lhs-rhs

    def solve_S_cri(self,e_arr):
        S_arr = np.linspace(0.1,1.6,100)
        S_sol = np.zeros([len(e_arr),len(S_arr)])
        S_cri = np.zeros(len(e_arr))
        for n_e,e in enumerate(e_arr):
            for n_S,S in enumerate(S_arr):
                S_sol[n_e,n_S] = self.energy_balance(S,e,alpha_i=self.ai)
            if np.min(np.abs(S_sol[n_e,:]))>1e-2*self.S0:
                S_cri[n_e] = np.nan
            else:
                S_cri[n_e] = S_arr[np.argmin(np.abs(S_sol[n_e,:]))]
        return S_cri

# ------------------------------------
# Streamlit Application
# ------------------------------------

st.title("Interactive Critical Insolation Visualization")

# Sidebar sliders for parameters in __init__
ao = st.sidebar.slider("ao (Open ocean albedo)", 0.0, 1.0, 0.4, 0.01)
ai = st.sidebar.slider("ai (Ice albedo)", 0.0, 1.0, 0.6, 0.01)
phi_c_deg = st.sidebar.slider("phi_c_deg (Collatitude in degrees)", 0.0, 90.0, 20.0, 1.0)
Fs = st.sidebar.slider("Fs (Solar constant scaling)", 100.0, 2000.0, 1360.0/np.pi, 10.0)
A = st.sidebar.slider("A (OLR constant term)", -300.0, -100.0, -218.4492, 1.0)
B = st.sidebar.slider("B (OLR linear term)", 0.5, 3.0, 1.4364, 0.01)
T_f = st.sidebar.slider("T_f (Freezing temp)", 250.0, 300.0, 280.0, 1.0)
k_i = st.sidebar.slider("k_i (Ice thermal conductivity)", 0.5, 5.0, 2.0, 0.1)
adv_flux = st.sidebar.slider("adv_flux (Advection flux)", 0.0, 1.0, 0.05, 0.01)

# Instantiate the class with updated parameters
SB = Critical_Insolation(ao=ao, ai=ai, phi_c_deg=phi_c_deg, Fs=Fs, A=A, B=B, T_f=T_f, k_i=k_i, adv_flux=adv_flux)
S_ice_free = SB.F_s_ice_free()
e_arr = np.linspace(0.005,0.4,50)
S_cri = SB.solve_S_cri(e_arr)

# Provided data (constants from your code)
S_HT_low = np.array([0.85, 0.87, 0.87, 0.87, 0.85,0.8])
S_HT_up= np.array([0.87, 0.9, 0.9, 0.9, 0.87,0.82])
e_SB = np.array([0,0.05,0.10,0.15,0.20,0.30])

S_SB_low = np.array([0.98, 0.98, 0.95, 0.9, 0.87,0.8])
S_SB_up = np.array([1, 0.99, 0.97, 0.92, 0.9,0.82])

fig,ax = plt.subplots(1,1,figsize=(12,8))

e_ice = e_arr

hd1, = plt.plot(e_ice,S_cri,label='Analytical Sol ($\\alpha_i={:.2f}$)'.format(ai))
plt.fill_between(e_ice,S_cri,np.zeros(len(e_ice)),facecolor='tab:blue',alpha=0.2)

hd2 = plt.errorbar(e_SB,(S_SB_low+S_SB_up)/2,yerr=(S_SB_up-S_SB_low)/2,label='GCM',marker='o',ls='none',c='tab:blue')
hd4 = plt.axhline(S_ice_free,c='tab:red')

y1 = S_cri
y2 = np.ones(len(S_cri))*S_ice_free
hd3 = plt.fill_between(e_ice,y1,y2, where=(y2 < y1),facecolor='None',alpha=0.3,hatch='//')

hd5 = plt.errorbar(e_SB,(S_HT_low+S_HT_up)/2,yerr=(S_HT_up-S_HT_low)/2,label='GCM',marker='o',ls='none',c='tab:red')

plt.legend([hd1,hd2,hd4,hd5,hd3],[
    'Analytical Sol for SB-start',
    'GCM',
    'Critical S for Ice-free start',
    'GCM',
    'Snowball Bistability'],loc='upper right')

plt.ylim(0.1,1.1)
plt.xlim(0,1.0)
plt.ylabel('Critical Annual Mean Insolation ($S_*$)')
plt.xlabel('Eccentricity')

st.pyplot(fig)
