import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import curve_fit

plt.rc('text', usetex=True)
t0, RMG0, URL0, IRL0  = np.genfromtxt('A3_1.txt', unpack=True) # s | ohm | mV | A
t1, RMG1, URL1, IRL1  = np.genfromtxt('A3b_1.txt', unpack=True) # s | ohm | mV | A
t2, RMG2, URL2, IRL2  = np.genfromtxt('A3b_2.txt', unpack=True) # s | ohm | mV | A

IMG = 10e-6     # muA, Strom am Messgerät


#t2,U2 = np.genfromtxt('A2_4.txt', unpack=True)
#t3,U3 = np.genfromtxt('A2_7.txt', unpack=True)
#Rkrit = np.empty(3)
#tkrit = np.empty(3)

# Zeitpunkt, an dem Magnet absenkt ist
# Werte abschneiden, Zeit bei Null beginnen lassen und U_krit bestimmen

cut = np.array([370,250,140])
for i in range(0,3):
    locals()["t"+str(i)]      =   locals()["t"+str(i)][cut[i]::]
    locals()["RMG"+str(i)]    = locals()["RMG"+str(i)][cut[i]::]
    locals()["IRL"+str(i)]    = locals()["IRL"+str(i)][cut[i]::]
    locals()["URL"+str(i)]    = locals()["URL"+str(i)][cut[i]::]
    locals()["t"+str(i)]      -= min(locals()["t"+str(i)])


def R(U,I):
    return U/I

# Kalibrierungsdaten des Platin-Temp-Sensors

a0  = 19.7443
da0 = 2.50532
a1  = 3.20409
da1 = 0.33372
a2  = -0.02161
da2 = 0.01102
a3  = 1.72736*1e-4
da3 = 9.85867*1e-5

def T(R):
    return a0 + a1*R + a2*R*R + a3*R*R*R

# irgendwie ist der Kalibrierungsfit nicht so schön, da hier riesige Fehler rauskommen, naja..
def dT(R):
    return np.sqrt((da0)**2 + (U*da1)**2 + (U*U*da2)**2 + (U*U*U*da3)**2)



# Plot R(T)
tkrit = 24
print("Tkrit liegt bei", T(RMG0[tkrit]))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlim([77, 150])
ax1.set_ylim([-0.2,11])
ax2 = ax1.twiny()

ax1.plot(T(RMG0),R(URL0,IRL0),".", label="Messung", color="#606da4",zorder=1)

ax2.tick_params(axis='x', labelcolor='#e5961c')
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks([T(RMG0[tkrit])])
ax2.set_xticklabels([round(T(RMG0[tkrit]),1)])

ax2.plot(np.ones(20)*T(RMG0[tkrit]), np.linspace(-0.5,12,20), "r--", linewidth=0.5, color="#e5961c", zorder=2)
ax2.text(T(RMG0[tkrit])+0.2, 10, s=r"T$_{krit}$", color="#e5961c", fontsize=12)
plt.annotate(s=str(round(R(URL0[tkrit],0.6),2))+r"$\,$m$\Omega$", xy=(T(RMG0[tkrit]),R(URL0[tkrit],0.6)), xytext=(T(RMG0[tkrit])+5,R(URL0[tkrit],0.6)+0.3), arrowprops=dict(color="black", arrowstyle="simple"))
ax1.legend(loc="best")
ax1.set_xlabel("Temperatur T / K")
ax1.set_ylabel(r"Widerstand R$_{SL}$ / m$\Omega$")
plt.savefig("R_T.pdf")
plt.clf()


# Plot R(T) für einen Abstand von 16mm
#plt.plot(t1,RMG1,"r.")
tkrit = 43
print("bei einem Abstand von 16mm liegt Tkrit bei", T(RMG1[tkrit]))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlim([77, 150])
ax1.set_ylim([-0.2,11])

ax2 = ax1.twiny()

ax1.plot(T(RMG1),R(URL1,IRL1),".", label=r"16$\,$mm Abstand", color="#606da4",zorder=1)

ax2.tick_params(axis='x', labelcolor='#e5961c')
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks([T(RMG1[tkrit])])
ax2.set_xticklabels([round(T(RMG1[tkrit]),1)])

ax2.plot(np.ones(20)*T(RMG1[tkrit]), np.linspace(-0.5,12,20), "r--", linewidth=0.5, color="#e5961c", zorder=2)
ax2.text(T(RMG1[tkrit])+0.2, 10, s=r"T$_{krit}$", color="#e5961c", fontsize=12)
plt.annotate(s=str(round(R(URL1[tkrit],0.6),2))+r"$\,$m$\Omega$", xy=(T(RMG1[tkrit]),R(URL1[tkrit],0.6)), xytext=(T(RMG1[tkrit])-10,R(URL1[tkrit],0.6)+1), arrowprops=dict(color="black", arrowstyle="simple"))
ax1.legend(loc="best")
ax1.set_xlabel("Temperatur T / K")
ax1.set_ylabel(r"Widerstand R$_{SL}$ / m$\Omega$")
plt.savefig("R_T_16mm.pdf")
plt.clf()


# Plot R(T) für einen Abstand von 10mm
#plt.plot(t2,RMG2,"r.")
tkrit = 49
print("bei einem Abstand von 10mm liegt Tkrit bei", T(RMG2[tkrit]))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlim([77, 150])
ax1.set_ylim([-0.2,11])
ax2 = ax1.twiny()

ax1.plot(T(RMG2),R(URL2,IRL2),".", label=r"10$\,$mm Abstand", color="#606da4",zorder=1)

ax2.tick_params(axis='x', labelcolor='#e5961c')
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks([T(RMG2[tkrit])])
ax2.set_xticklabels([round(T(RMG2[tkrit]),1)])

ax2.plot(np.ones(20)*T(RMG2[tkrit]), np.linspace(-0.5,12,20), "r--", linewidth=0.5, color="#e5961c", zorder=2)
ax2.text(T(RMG2[tkrit])+0.2, 10, s=r"T$_{krit}$", color="#e5961c", fontsize=12)
plt.annotate(s=str(round(R(URL2[tkrit],0.6),2))+r"$\,$m$\Omega$", xy=(T(RMG2[tkrit]),R(URL2[tkrit],0.6)), xytext=(T(RMG2[tkrit])-10,R(URL2[tkrit],0.6)+1), arrowprops=dict(color="black", arrowstyle="simple"))
ax1.legend(loc="best")
ax1.set_xlabel("Temperatur T / K")
ax1.set_ylabel(r"Widerstand R$_{SL}$ / m$\Omega$")
plt.savefig("R_T_10mm.pdf")
