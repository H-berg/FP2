import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import curve_fit

plt.rc('text', usetex=True)
t0, RMG0, URL0, IRL0  = np.genfromtxt('A3_1.txt', unpack=True) # s | ohm | mV | A
t1, RMG1, URL1, IRL1  = np.genfromtxt('A3b_1.txt', unpack=True) # s | ohm | mV | A
t2, RMG2, URL2, IRL2  = np.genfromtxt('A3b_2.txt', unpack=True) # s | ohm | mV | A

IMG = 10e-6     # muA, Strom am Messgerät

# Zeitpunkt, an dem Magnet absenkt ist
# Werte abschneiden, Zeit bei Null beginnen lassen und U_krit bestimmen

cut = np.array([300,250,140])
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

    #for k in range(1,3):
    #    dTmittel[i-1] += np.abs(T(RMG[TK])-T(RMG[TK-k]))
    #    dTmittel[i-1] += np.abs(T(RMG[TK])-T(RMG[TK+k]))
    #dTmittel[i-1] /= 2*k
    #print("Tempaenderung beim Ablesen = ", dTmittel[i-1], " K")
    #print()



# Plot R(T)
tkrit = 94
dTmittel = 0
# Temperaturänderung im Ablesebereich zur Fehlerabschätzung
for k in range(1,3):
    dTmittel += np.abs(T(RMG0[tkrit])-T(RMG0[tkrit-k]))
    dTmittel += np.abs(T(RMG0[tkrit])-T(RMG0[tkrit+k]))
dTmittel /= 2*k
dTmittel += 2       # abgeschätzter systematische Fehler

print("Tkrit liegt bei", T(RMG0[tkrit]),"+-", dTmittel, "K")

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlim([77, 150])
ax1.set_ylim([-0.2,11])
ax2 = ax1.twiny()

ax1.plot(T(RMG0),R(URL0,IRL0),".", color="#606da4",zorder=1)
#ax1.errorbar(T(RMG0),R(URL0,IRL0), xerr=dTmittel, capsize=3, fmt='.', label="Messung", color="#606da4", zorder=1)

ax2.tick_params(axis='x', labelcolor='#e5961c')
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks([T(RMG0[tkrit])])
ax2.set_xticklabels([round(T(RMG0[tkrit]),1)])

ax2.plot(np.ones(20)*T(RMG0[tkrit]), np.linspace(-0.5,12,20), "r--", linewidth=0.5, color="#e5961c", zorder=2)
ax2.text(T(RMG0[tkrit])+0.2, 10, s=r"T$^{4MP}_{krit}$", color="#e5961c", fontsize=12)
plt.annotate(s=str(round(R(URL0[tkrit],0.6),2))+r"$\,$m$\Omega$", xy=(T(RMG0[tkrit]),R(URL0[tkrit],0.6)), xytext=(T(RMG0[tkrit])+5,R(URL0[tkrit],0.6)+0.3), arrowprops=dict(color="black", arrowstyle="simple"))
#ax1.legend(loc="best")
ax1.set_xlabel("Temperatur T / K")
ax1.set_ylabel(r"Widerstand R$_{SL}$ / m$\Omega$")
plt.savefig("R_T.pdf")
plt.clf()


# Plot R(T) für einen Abstand von 16mm
#plt.plot(t1,RMG1,"r.")
tkrit = 43
dTmittel = 0
# Temperaturänderung im Ablesebereich zur Fehlerabschätzung
for k in range(1,3):
    dTmittel += np.abs(T(RMG1[tkrit])-T(RMG1[tkrit-k]))
    dTmittel += np.abs(T(RMG1[tkrit])-T(RMG1[tkrit+k]))
dTmittel /= 2*k
dTmittel += 2       # abgeschätzter systematische Fehler

print("bei einem Abstand von 16mm liegt Tkrit bei", T(RMG1[tkrit]),"+-", dTmittel, "K")

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
ax2.text(T(RMG1[tkrit])+0.2, 10, s=r"T$^{4MP}_{krit}$", color="#e5961c", fontsize=12)
plt.annotate(s=str(round(R(URL1[tkrit],0.6),2))+r"$\,$m$\Omega$", xy=(T(RMG1[tkrit]),R(URL1[tkrit],0.6)), xytext=(T(RMG1[tkrit])-10,R(URL1[tkrit],0.6)+1), arrowprops=dict(color="black", arrowstyle="simple"))
ax1.legend(loc="best")
ax1.set_xlabel("Temperatur T / K")
ax1.set_ylabel(r"Widerstand R$_{SL}$ / m$\Omega$")
plt.savefig("R_T_16mm.pdf")
plt.clf()


# Plot R(T) für einen Abstand von 10mm
# Boltzmann-Fit weil die Messung etwas beschissen ist
def boltz(x, a,b,c,d):
    y  = a-b
    y /= 1 + np.exp((x - d)/c)
    y += b
    return y

# Werte zum Fitten präparieren: nur Werte welche in einem schlauch liegen.
p1 = np.array([-0.1,7,4,120])
W       = R(URL2,IRL2)[0:150]
TEMP    = T(RMG2)[0:150]

j = np.where(W > boltz(TEMP, *p1))
W       = W[j]
TEMP    = TEMP[j]

p2      = np.array([0.1,10,4,120])
j = np.where(W < boltz(TEMP, *p2))
W       = W[j]
TEMP    = TEMP[j]

# fit
p0      = np.array([0,8,0.4,120])
bounds  = ([-1,0,0,0],[1,20,10,200])
# offset korrektur
params, covariance_matrix = curve_fit(boltz, TEMP, W, p0, bounds=bounds)
offset       = params[0]
W           -= offset

params, covariance_matrix = curve_fit(boltz, TEMP, W, p0, bounds=bounds)
uncertainties = np.sqrt(np.diag(covariance_matrix))

print("Boltz-Fkt:")
for name, value, uncertainty in zip('abcd', params, uncertainties):
    print(f'{name} = {value:8.3f} ± {uncertainty:.3f}')
    exec("%s = %f" % (name,value))
    exec("%s = %f" % ("d"+name, uncertainty))

# Temperatur bei 0.05mOhm durch probieren
T005 = 99.8

x = np.linspace(78,150,1000)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlim([77, 150])
ax1.set_ylim([-0.2,11])
ax2 = ax1.twiny()

ax1.plot(TEMP, W, "x",color="#606da4", label="Für den Fit berücksichtigt")
ax1.plot(T(RMG2),R(URL2,IRL2)-offset,".", label=r"10$\,$mm Abstand", color="#606da4",zorder=1)
#ax1.plot(T(RMG2[0:80]), boltz(T(RMG2[0:80]), *params))
ax1.plot(x, boltz(x, *p1),"r-.", linewidth = 0.5, zorder=1)     #händisch annähern
ax1.plot(x, boltz(x, *p2),"r-.", linewidth = 0.5, zorder=1)     #händisch annähern
#ax1.plot(x, boltz(x, *p0), label="p0")     #händisch annähern
ax1.plot(x, boltz(x, *params), linewidth = 2, color="#acb2ba", label="Fit")

ax2.tick_params(axis='x', labelcolor='#e5961c')
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks([T005])
ax2.set_xticklabels([round(T005,1)])

ax2.plot(np.ones(20)*T005, np.linspace(-0.5,12,20), "r--", linewidth=0.5, color="#e5961c", zorder=2)
ax2.text(T005+0.2, 10, s=r"T$^{4MP}_{krit}$", color="#e5961c", fontsize=12)
plt.annotate(s=str(round(boltz(T005,*params),2))+r"$\,$m$\Omega$", xy=(T005,0.05), xytext=(T005-10,0.05+1.5), arrowprops=dict(color="black", arrowstyle="simple"))
ax1.legend(loc=1)
ax1.set_xlabel("Temperatur T / K")
ax1.set_ylabel(r"Widerstand R$_{SL}$ / m$\Omega$")
plt.savefig("R_T_10mm.pdf")
