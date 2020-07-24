import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Auslese
def auslesen(file):
    x = np.genfromtxt(file, unpack=True)
    x[0] *= 1e3      # in millisek
    # Offset weg
    x[5]        -= min(x[5])
    # Normierung
    x[5]        /= max(x[5])
    return x[0], x[5]

# ----------------------- Plot ------------------------------------------------
def plot(x,y,file,func, params):
    t = np.linspace(min(x),max(x), 100000)
    plt.figure(figsize=(10,5))
    plt.plot(x,y, "r.", label="Messwerte")
    plt.plot(t, func(t, *params), label = "S2-Fit-Funktion")

    plt.xscale("log")
    plt.xlabel(r"Mischzeit t$_m$(ms)")
    plt.ylabel("Amplitude (stimuliertes Echo)")
    plt.axis([min(x)-0.002,max(x)+100,-0.05, 1.05])
    plt.legend()
    plt.savefig("Abb/"+file+".pdf", dpi = 1000)
    plt.close()


# -------------------- T1 - Zeit bestimmen ------------------------------------
def exp_T1(t,A,b,T):
    return A*np.exp(-(t/T)**b)

def T1(file):
    time, real_off   = auslesen(file)
    p0 = np.array([max(real_off), 2.5,14])   #Schätzwerte
    bounds = ([0,-np.inf,0],[1,np.inf,100])   # Intervall
    params, covariance_matrix = curve_fit(exp_T1, time, real_off, p0, bounds=bounds)
    uncertainties = np.sqrt(np.diag(covariance_matrix))
    plot(time,real_off,file, exp_T1, params)
    return params[2], uncertainties[2]

# ---------------------------cos-cos-------------------------------------------
def coscos(file, T1):
    def exp_cos(t,S0,A,B,b,c,tau):
        return S0 + (A*np.exp(-(t/tau)**b)+B)*np.exp(-(t/T1)**c)

    time, real_off  = auslesen(file)
    # Fit
    p0 = np.array([0.2,0.5, 0.3,1.5, 1.5, 1])   #Schätzwerte
    bounds = ([0,0,0,-np.inf,-np.inf,0],[2,2,2,np.inf,np.inf,T1])   # Intervall
    params, covariance_matrix = curve_fit(exp_cos, time, real_off, p0, bounds=bounds)

    uncertainties = np.sqrt(np.diag(covariance_matrix))
    #print(params)
    plot(time,real_off,file, exp_cos, params)
    return params[5], uncertainties[5]


# ----------------------------sin-sin------------------------------------------
def exp_sin(t,S0,A,B,b,c,tau,T1):
    return S0 + (A*np.exp(-(t/tau)**b)+B)*np.exp(-(t/T1)**c)

def sinsin(file):
    time, real_off  = auslesen(file)
    # Fit
    p0 = np.array([0.2,0.5, 0.3,1.5, 1.5, 1,14])   #Schätzwerte
    bounds = ([0,0,0,-np.inf,-np.inf,0,0],[2,1,1,np.inf,np.inf,20,100])   # Intervall
    params, covariance_matrix = curve_fit(exp_sin, time, real_off, p0, bounds=bounds)

    uncertainties = np.sqrt(np.diag(covariance_matrix))
    plot(time,real_off,file, exp_sin, params)
    return params[5], uncertainties[5], params[6], uncertainties[6]



with open('data.txt', 'a') as f:
    f.truncate(0)
    np.savetxt(f, [], header='Temp. T1 ∆T1 Tau_cos ∆Tau_cos Tau_sin ∆Tau_sin T1_sin ∆T1_sin')
    for i in np.arange(310, 349, 3):
        t1, t1_err = T1('T1/DMSO2_T1_' + str(i) + 'K.dat.nmr')
        tau_c, tau_c_err = coscos('cos/DMSO2_F2_' + str(i) + 'K.dat.nmr', t1)
        tau_s, tau_s_err, t1_s, t1_s_err = sinsin('sin/DMSO2_F2_' + str(i) + 'K.dat.nmr')


        np.savetxt(f, np.c_[i, t1, t1_err, tau_c, tau_c_err, tau_s, tau_s_err, t1_s, t1_s_err], fmt='%.4f')
#print(auslesen('T1.dat.nmr', [0,1,2,3]))
