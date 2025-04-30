import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
# 8th order method
def RKDP8_method(f,y0,t):
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(0,len(t)-1):
        h = t[i+1]-t[i]
        k1 = h*f(y[i],t[i])
        k2 = h*f(y[i],t[i])
        k3 = h*f(y[i],t[i])
        k4 = h*f(y[i],t[i])
        k5 = h*f(y[i],t[i])
        k6 = h*f(y[i],t[i])
        k7 = h*f(y[i],t[i])
        k8 = h*f(y[i],t[i])
        k9 = h*f(y[i],t[i])
        k10 = h*f(y[i],t[i])
        k11 = h*f(y[i],t[i])
        y[i+1] = y[i] + k1*(1/20) + k9*(16/45) + k10*(49/180) + k11*(1/20) + k7*(- 49/180)
    return y

# 5th order methods
def RKDP_method(f,y0,t):
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(0,len(t)-1):
        h = t[i+1]-t[i]
        k1 = h*f(y[i], t[i]+h*(0))
        k2 = h*f(y[i]+k1*(1/5), t[i]+h*(1/5))
        k3 = h*f(y[i]+k1*(3/40)+k2*(9/40), t[i]+h*(3/10))
        k4 = h*f(y[i]+k1*(44/45)+k2*(-56/15)+k3*(32/9), t[i]+h*(4/5))
        k5 = h*f(y[i]+k1*(19372/6561)+k2*(-25360/2187)+k3*(64448/6561)+k4*(-212/729), t[i]+h*(8/9))
        k6 = h*f(y[i]+k1*(9017/3168)+k2*(-355/33)+k3*(46732/5247)+k4*(49/176)+k5*(-5103/18656), t[i]+h*(1))
        k7 = h*f(y[i]+k1*(35/384)+k2*(0)+k3*(500/1113)+k4*(125/192)+k5*(-2187/6784)+k6*(11/84), t[i]+h*(1))
        y[i+1] = y[i] + (5179/57600)*(k1) + 0*k2 + (7571/16695)*k3 + (393/640)*k4 + (-92097/339200)*k5 + (187/2100)*k6 + (1/40)*k7
    return y
def Nymstrom5_method(f,y0,t):
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(0,len(t)-1):
        h = t[i+1]-t[i]
        k1 = h*f(y[i],t[i])
        k2 = h*f(y[i]+k1/3,t[i]+h/3)
        k3 = h*f((y[i]+k1*(4/25)+k2*(6/25)),(t[i]+h*(2/5)))
        k4 = h*f((y[i]+k1*(1/4)+k2*(-3)+k3*(15/4)),(t[i]+h))
        k5 = h*f((y[i]+k1*(2/27)+k2*(10/9)+k3*(-50/81)+k4*(8/81)),(t[i]+h*(2/3)))
        k6 = h*f((y[i]+k1*(2/25)+k2*(12/25)+k3*(2/15)+k4*(8/75)),(t[i]+h*(4/5)))
        y[i+1] = y[i] + (23/192)*(k1) + (0)*k2 + (125/192)*k3 + 0*k4 + (-27/64)*k5 + (125/192)*k6
    return y

# Fourth order methods
def RK4_method(f,y0,t):
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(0,len(t)-1):
        h = t[i+1]-t[i]
        k1 = h*f(y[i],t[i])
        k2 = h*f((y[i]+k1/2),(t[i]+h/2))
        k3 = h*f((y[i]+k2/2),(t[i]+h/2))
        k4 = h*f((y[i]+k3),(t[i]+h))
        y[i+1] = y[i] + 1/6*(k1 + 2*k2 + 2*k3 + k4)
    return y
def ThreeEigthsRule_method(f,y0,t):
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(0,len(t)-1):
        h = t[i+1]-t[i]
        k1 = h*f(y[i],t[i])
        k2 = h*f((y[i]+k1/3),(t[i]+h/3))
        k3 = h*f((y[i]-k1/3+k2),(t[i]+h*(2/3)))
        k4 = h*f((y[i]+k1-k2+k3),(t[i]+h))
        y[i+1] = y[i] + 1/8*(k1 + 3*k2 + 3*k3 + k4)
    return y

# Third order methods
def RK3_method(f,y0,t):
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(0,len(t)-1):
        h = t[i+1]-t[i]
        k1 = h*f(y[i],t[i])
        k2 = h*f((y[i]+k1/2),(t[i]+h/2))
        k3 = h*f((y[i]-k1+k2*2),(t[i]+h))
        y[i+1] = y[i] + 1/6*(k1 + 4*k2 + k3)
    return y
def Huens3_method(f,y0,t):
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(0,len(t)-1):
        h = t[i+1]-t[i]
        k1 = h*f(y[i],t[i])
        k2 = h*f((y[i]+k1/3),(t[i]+h*(1/3)))
        k3 = h*f((y[i]+k2*(2/3)),(t[i]+h*(2/3)))
        y[i+1] = y[i] + (1/4)*(k1 + 3*k3)
    return y
def Ralstons3_method(f,y0,t):
    y= np.zeros(len(t))
    y[0] = y0
    for i in range(0,len(t)-1):
        h = t[i+1]-t[i]
        k1 = h*f(y[i],t[i])
        k2 = h*f((y[i]+k1/2),(t[i]+h/2))
        k3 = h*f((y[i]+k2*(3/4)),(t[i]+h*(3/4)))
        y[i+1] = y[i] + (1/9)*(2*k1 + 3*k2 + 4*k3)
    return y
def VanderHouwe3_method(f,y0,t):
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(0,len(t)-1):
        h = t[i+1]-t[i]
        k1 = h*f(y[i],t[i])
        k2 = h*f((y[i]+k1*(8/15)),(t[i]+h*(8/15)))
        k3 = h*f((y[i]+k1/4+k2*(5/12)),(t[i]+h*(2/3)))
        y[i+1] = y[i] + 1/4*(k1 + 3*k3)
    return y
def SSPRK3_method(f,y0,t):
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(0,len(t)-1):
        h = t[i+1]-t[i]
        k1 = h*f(y[i],t[i])
        k2 = h*f((y[i]+k1),(t[i]+h))
        k3 = h*f((y[i]+k1/4+k2/4),(t[i]+h/2))
        y[i+1] = y[i] + 1/6*(k1 + k2 + 4*k3)
    return y

# Second order methods
def Ralstons_method(f,y0,t):
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(0,len(t)-1):
        h = t[i+1]-t[i]
        k1 = h*f(y[i],t[i])
        k2 = h*f((y[i]+(2/3)*k1),(t[i]+(2/3)*h))
        y[i+1] = y[i] + (1/4)*k1 + (3/4)*k2
    return y
def Heuns_method(f,y0,t):
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(0,len(t)-1):
        h = t[i+1]-t[i]
        k1 = h*f(y[i],t[i])
        k2 = h*f((y[i]+k1),(t[i]+h))
        y[i+1] = y[i] + 1/2*(k1 + k2)
    return y
def Midpoint_method(f,y0,t):
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(0,len(t)-1):
        h = t[i+1]-t[i]
        k1 = h*f(y[i],t[i])
        k2 = h*f((y[i]+k1*(1/2)),(t[i]+(1/2)*h))
        y[i+1] = y[i] + k1*0 + k2
    return y

# First order method
def Euler_method(f,y0,t):
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(0,len(t)-1):
        h = t[i+1]-t[i]
        y[i+1] = y[i] + h*f(y[i],t[i])
    return y

# Differential equation
def f(y,t):
    return 1*(np.sin(t))**2 * y

y0 = 1 # Initial condition
t = np.linspace(0,5,10) # Time grid

# 5th order methods
y_RKDP = RKDP_method(f,y0,t)
y_Nymstrom5 = Nymstrom5_method(f,y0,t)

# 4th order methods
y_RK4 = RK4_method(f,y0,t)
y_38 = ThreeEigthsRule_method(f,y0,t)

# 3rd order methods
y_RK3 = RK3_method(f,y0,t)
y_Huens3 = Huens3_method(f,y0,t)
y_Ralstons3 = Ralstons3_method(f,y0,t)
y_VanderHouwe3 = VanderHouwe3_method(f,y0,t)
y_SSPRK3 = SSPRK3_method(f,y0,t)

# 2nd order methods
y_Ralstons = Ralstons_method(f,y0,t)
y_Heuns = Heuns_method(f,y0,t)
y_Midpoint = Midpoint_method(f,y0,t)

# 1st order method
y_Euler = Euler_method(f,y0,t)

# Call RK with a finer time grid for actual 
t_fine = np.linspace(0,5,100000) # Time grid
y_actual = np.exp(1/2 * (t_fine - np.sin(2 * t_fine) * (1/2)))  # Actual solution
# Plot the results
plt.plot(t_fine, y_actual, label="Actual Solution")
plt.plot(t, y_RKDP, label="RKDP")
plt.plot(t, y_Nymstrom5, label="Nymstrom5")
plt.plot(t, y_RK4, label="RK4")
plt.plot(t, y_RK3, label="RK3")
plt.plot(t, y_38, label="3/8 Rule")
plt.plot(t, y_Huens3, label="Huen's3")
plt.plot(t, y_Ralstons3, label="Ralston's3")
plt.plot(t, y_VanderHouwe3, label="VanderHouwen/Wray's3")
plt.plot(t, y_SSPRK3, label="SSPRK3")
plt.plot(t, y_Ralstons, label="Ralston's")
plt.plot(t, y_Heuns, label="Huens")
plt.plot(t, y_Midpoint, label="Midpoint")
plt.plot(t, y_Euler, label="Euler")
plt.xlabel("Time t")
plt.ylabel("y(t)")
plt.title("Comparison of Numerical Methods")
plt.legend()
plt.grid()
plt.show()

"""
# Interpolate y_actual onto the coarser grid t
y_actual_interp = interp1d(t_fine, y_actual, kind='linear')(t)

# Compute the delta (absolute difference) for each method
delta_RKDP = np.abs(y_actual_interp - y_RKDP)
delta_Nymstrom5 = np.abs(y_actual_interp - y_Nymstrom5)
delta_RK4 = np.abs(y_actual_interp - y_RK4)
delta_RK3 = np.abs(y_actual_interp - y_RK3)
delta_38 = np.abs(y_actual_interp - y_38)
delta_Huens3 = np.abs(y_actual_interp - y_Huens3)
delta_Ralstons3 = np.abs(y_actual_interp - y_Ralstons3)
delta_VanderHouwe3 = np.abs(y_actual_interp - y_VanderHouwe3)
delta_SSPRK3 = np.abs(y_actual_interp - y_SSPRK3)
delta_Ralstons = np.abs(y_actual_interp - y_Ralstons)
delta_Heuns = np.abs(y_actual_interp - y_Heuns)
delta_Midpoint = np.abs(y_actual_interp - y_Midpoint)
delta_Euler = np.abs(y_actual_interp - y_Euler)

# Create a new plot for the delta functions
plt.figure()
plt.plot(t, delta_RKDP, label="Delta RKDP")
plt.plot(t, delta_Nymstrom5, label="Delta Nymstrom5")
plt.plot(t, delta_RK4, label="Delta RK4")
plt.plot(t, delta_RK3, label="Delta RK3")
plt.plot(t, delta_38, label="Delta 3/8 Rule")
plt.plot(t, delta_Huens3, label="Delta Huen's3")
plt.plot(t, delta_Ralstons3, label="Delta Ralston's3")
plt.plot(t, delta_VanderHouwe3, label="Delta VanderHouwen/Wray's3")
plt.plot(t, delta_SSPRK3, label="Delta SSPRK3")
plt.plot(t, delta_Ralstons, label="Delta Ralston's")
plt.plot(t, delta_Heuns, label="Delta Huens")
plt.plot(t, delta_Midpoint, label="Delta Midpoint")
plt.plot(t, delta_Euler, label="Delta Euler")

plt.yscale('log')
plt.xlabel("Time t")
plt.ylabel("Delta (|y_actual - y_method|)")
plt.title("Delta Functions: Numerical Methods vs Actual Solution")
plt.legend()
plt.grid()
plt.show()
"""