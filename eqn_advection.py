import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
plt.rcParams["figure.autolayout"] = True
from lib_sdf import ASEC, ASIC, ASEDA, ASLF, ASLW


def cin1(x):
    return (np.cos(np.pi*x))**8

# paramètre du schéma
V = 3
t_max = 1

cini   = "u_0(x)"
cistr1 = "cos(\pi*x)^8"

cin_data = {"CI1": [cin1,cistr1]}

#condition initiale à utiliser :
""" Saisir CI = "CI1" ou "CI2" pour respectivement utiliser sin(pi*x) ou exp(...)"""
CI = "CI1" #"CI2"

#Nombre de courbes à afficher -1
nb_curves = 20
# dictionnaire des schémas 
d_metho = {"Schéma Explicite centré ": ASEC,"Schéma Implicite centré ": ASIC, "Schéma Explicite décentré amont ": ASEDA,
           "Schéma de Lax Friedrich ": ASLF,"Schéma de Lax Wendroff ": ASLW}
            
metho_on = [True, True, True, True, True]
# couple (h,tau) à tester
l_tests = [(0.04,0.01)]#,(0.01,0.002),(0.005,0.004),(0.005,0.0002)]

i = 0

for k,f in d_metho.items():
   
    if metho_on[i] == True:
        print(f"Méthode : {k}")
        for e in l_tests:
            h = e[0]
            tau = e[1]
            N = round(1/h)

            """Besoin d'un pas en condition de stabilité !!"""
            X,T,U = f(N,tau,cin_data[CI][0], V, t_max)

            S = np.empty((len(T),len(X)))
            delta = np.zeros((len(T),len(X)))

            """ # Comparaison avec solution exacte
            if CI == "CI1":
                for ti in range(len(T)):
                    for xi in range(len(X)):
                        sol = np.sin(np.pi*X[xi])*np.cos(np.pi*T[ti])
                        S[ti,xi] = sol
                        delta[ti,xi] = np.abs(sol-U[ti,xi])
            maxdelta = max(delta.ravel()) 
            print(f'max erreur {k} : ',maxdelta)           
            """           
            liste_t = np.linspace(0,t_max,nb_curves+1)
            fig, ax = plt.subplots(figsize=(20, 12), dpi=80)
            
            for t in liste_t:
                n = round(t/tau)
                ax.plot(X,U[n,:],label = 't ='+str(round(t,2)))
                #ax.plot(X,S[n,:],label = 'S t ='+str(round(t,2)))
                ax.legend(loc = "upper right")
                plt.draw()
                # plt.pause(1)
            plt.xlabel(u'$X$', fontsize=18)
            plt.ylabel(u'$U_n$', fontsize=18, rotation=0)
            plt.title(fr"Propagation de l'advection' pour $h = {e[0]}$ et $\tau = {e[1]}$"
                f"\npour la méthode {k}"
                f"\net la condition initiale ${cini} = {cin_data[CI][1]}$")
            plt.show()
    i += 1