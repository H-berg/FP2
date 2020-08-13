import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Auslese
def auslesen(file):
    x = np.genfromtxt(file, unpack=True)
    x[0] *= 1e3      # in millisek
    y = y = np.sqrt(x[5]**2 + x[7]**2)
    # Offset weg
    y           -= min(y)
    # Normierung
    y           /= max(y)
    return x[0], y

# ----------------------- Plot ------------------------------------------------
def plot(x,y,file,func, params):
    t = np.linspace(min(x),max(x), 100000)
    plt.figure(figsize=(10,5))
    plt.plot(x,y, "r.", label="Messwerte")
    #plt.plot(t, func(t, *params), label = "S2-Fit-Funktion")

    plt.xscale("log")
    plt.xlabel(r"Mischzeit t$_m$(ms)")
    plt.ylabel(r"Signalintensität I ($|Re + Im|$)")
    plt.axis([min(x)-0.002,1.2,-0.05, 1.05])
    plt.legend()
    plt.savefig("Abb/"+file+".pdf", dpi = 1000)
    plt.close()


# -------------------- T2 - Zeit bestimmen ------------------------------------
def exp_T2(t,A,B,b,T2):
    return A + B*np.exp(-(t/T2)**b)

def T2(file):
    time, y   = auslesen(file)
    #p0 = np.array([max(y)+100, max(y),-2 ,0.2])   #Schätzwerte
    #bounds = ([max(y),0,-np.inf,0],[np.inf,np.inf,np.inf,5])   # Intervall
    #params, covariance_matrix = curve_fit(exp_T2, time, y, p0, bounds=bounds)
    #uncertainties = np.sqrt(np.diag(covariance_matrix))
    plot(time,y,file, exp_T2, 0)
    #return params[3], uncertainties[3]


with open('T1.txt', 'a') as f:
    f.truncate(0)
    np.savetxt(f, [], header='Temp. T2 ∆T2')
    for i in np.arange(310, 349, 3):
        T2('T2/DMSO2_T2_' + str(i) + 'K.dat.nmr')

        #np.savetxt(f, np.c_[i, t2, t2_err], fmt='%.4f')
