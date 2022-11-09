import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
plt.rcParams["figure.autolayout"] = True
from lib_sdf import OSEC,OSIC,OASI

def cin1(x):
    return (np.sin(np.pi*x))

def cin2(x):
    return (np.exp(-10*(3*x-1)**2))


# paramètre du schéma
gamma = 1
t_max = 1
v0 = 0

cini   = "u_0(x)"
cistr1 = "sin(\pi*x)"
cistr2 = "e^{-10(3x-1)^2}"

cin_data = {"CI1": [cin1,cistr1],"CI2": [cin2,cistr2]}

#condition initiale à utiliser :
""" Saisir CI = "CI1" ou "CI2" pour respectivement utiliser sin(pi*x) ou exp(...)"""
CI = "CI2"

#Nombre de courbes à afficher -1
nb_curves = 20

# dictionnaire des schémas
d_metho = {"Schéma Explicite Centré": OSEC,"Schéma Implicite Centré 1":OSIC, "Autre Schéma Implicite" :OASI}
# quel méthode on souhaite activer
metho_on = [True, True, True]
# couple (h,tau) à tester
l_tests = [(.04,.05),(.01,.005)]

i = 0

for k,f in d_metho.items():
    if metho_on[i] == True:
        print(f"Méthode : {k}")
        for e in l_tests:
            h = e[0]
            tau = e[1]
            N = round(1/h) - 1

            """Besoin d'un pas en condition de stabilité !!"""
            X,T,U = f(N,tau,cin_data[CI][0],gamma,t_max,v0)

            S = np.empty((len(T),len(X)))
            delta = np.zeros((len(T),len(X)))
            
            if CI == "CI1":
                for ti in range(len(T)):
                    for xi in range(len(X)):
                        sol = np.sin(np.pi*X[xi])*np.cos(np.pi*T[ti])
                        S[ti,xi] = sol
                        delta[ti,xi] = np.abs(sol-U[ti,xi])
            maxdelta = max(delta.ravel()) 
            print(f'max erreur {k} : ',maxdelta)           
                        
            liste_t = np.linspace(0,t_max,nb_curves+1)
            fig, ax = plt.subplots(figsize=(20, 12), dpi=80)

            for t in liste_t:
                n = round(t/tau)
                ax.plot(X,U[n,:],label = 't ='+str(round(t,2)))
                ax.plot(X,S[n,:],label = 'S t ='+str(round(t,2)))
                ax.legend(loc = "upper right")
                plt.draw()
                # plt.pause(1)
            plt.xlabel(u'$X$', fontsize=18)
            plt.ylabel(u'$U_n$', fontsize=18, rotation=0)
            plt.title(fr"Propagation d'onde obtenue pour $h = {e[0]}$ et $\tau = {e[1]}$"
                f"\npour la méthode {k}"
                f"\net la condition initiale ${cini} = {cin_data[CI][1]}$")
            plt.show()
    i += 1
