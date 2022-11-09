"""
TITRE : Librairie de fonction pour l'application de la méthode des éléments finis
AUTEUR : Maxime GOSSELIN


¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤
          EQUATION DES ONDES 
¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤

(∂^2 u)/(∂t^2 )-γ^2  (∂^2 u)/(∂x^2 )= 0

SCHEMAS : 

 - Explicite centré [OSEC], 
 - Implicite centré [OSIC]
 - Implicite à 7 points [OASI]

CONDITIONS AUX LIMITES : 

 - Dirichlet : u(0,t)=0 ; u(1,t)=0 ; u_0^n=0 ; u_(N+1)^n

¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤
       EQUATION DE LA CHALEUR 
¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤

(∂u)/(∂t)-ν(∂^2 u)/(∂x^2) = 0


SCHEMAS : 

 - Euler Explicite  [CSEE], 
 - Euler Implicite  [CSEI]
 - Cranck-Nicholson [CSCN]

CONDITIONS AUX LIMITES : 

 - Dirichlet : u(0,t)=0 ; u(1,t)=0 ; u_0^n=0 ; u_(N+1)^n

¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤
       EQUATION D'ADVECTION
¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤

(∂u)/(∂t) + 3(∂u)/(∂x) = 0 ;  0<= x <= 1

SCHEMAS : 

 - Explicite centré  [ASEC], 
 - Implicite centré  [ASIC]
 - Explicite décentré amont [ASEDA]
 - Lax Friedrich [ASLF]
 - Lax Wendroff [ASLW]

CONDITIONS AUX LIMITES : 

 - Périodique : u(x,t+1) = u(x,t)

"""

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve

#%%

def OSEC(N,tau,u0,gamma,Tmax,v0):
    """Schéma Explicite Centré"""
    
    #pas
    n_max = int(Tmax/tau)
    h = 1/(N+1)
    
    # espace considéré
    X = np.linspace(0,1,N+2)                #liste des x
    # temps considéré
    T = np.linspace(0,n_max*tau,n_max+1)    #liste des temps
    
    # Matrice des résultats
    U = np.zeros((n_max+1,N+2))             #dit explicitement
    U[0,:] = u0(X)
    U[1,:] = U[0,:]+tau*v0                  # condition initiale 2 d'ordre 1

    # condition initiale
    un = U[1,1:N+1]
    un1 = U[0,1:N+1]
    
    #constante
    c = gamma**2*tau**2/(h**2)
    
    #Matrices
    M1= np.zeros((N,N))
    M2 = np.eye(N)
    
    for i in range(N):
        M1[i,i] = 2*(1-c)
        
    for i in range(N-1):
        M1[i,i+1]= c
        M1[i+1,i]= c
    
    for n in range(2,n_max+1):
        """Pas besoin d'utiliser linalg solve car explicite"""

        unp1 = M1@un - M2@un1
        U[n,1:N+1] = unp1
        un1 = un
        un = unp1

    return X,T,U

#%%

def OSIC(N,tau,u0,gamma,Tmax,v0):
    """Schéma Implicite Centré"""
    #pas
    n_max = int(Tmax/tau)
    h = 1/(N+1)
    
    # espace considéré
    X = np.linspace(0,1,N+2)                #liste des x
    
    # temps considéré
    T = np.linspace(0,n_max*tau,n_max+1)    #liste des temps
    
    # Matrice des résultats
    U = np.zeros((n_max+1,N+2))             #dit explicitement
    U[0,:] = u0(X)
    U[1,:] = U[0,:]+tau*v0

    # condition initiale
    un = U[1,1:N+1]
    un1 = U[0,1:N+1]
    
    #constante
    c = gamma**2*tau**2/(h**2)
    
    #Matrices
    M1= np.zeros((N,N))
    M2 = np.eye(N)*2*c
    M3 = np.eye(N)

    for i in range(N):
        M1[i,i] = 1+2*c
        
    for i in range(N-1):
        M1[i,i+1]=-c
        M1[i+1,i]=-c

    for n in range(2,n_max+1):
        ui = M2@un - M3@un1
        unp1 = np.linalg.solve(M1,ui)
        U[n,1:N+1] = unp1
        un1 = un
        un = unp1

    return X,T,U

#%%

def OASI(N,tau,u0,gamma,Tmax,v0):
    """Autre Schéma Implicite"""
    h = 1/(N + 1)
    c = gamma*tau/h

    # temps considéré
    n_max = int(Tmax/tau)
    T = np.linspace(0,n_max*tau,n_max+1)

    # espace considéré
    X = np.linspace(0,1,N+2)
    
    # Matrice des résultats
    U = np.zeros((n_max+1,N+2)) 
    U[0,:] = u0(X)
    U[1,:] = U[0,:]+tau*v0

    # condition initiale
    un = U[1,1:N+1]
    un1 = U[0,1:N+1]
    
    # définition de M1
    M1 = np.zeros((N,N))
    for i in range(N):
        M1[i,i] = 1+c**2
    for i in range(N-1):
        M1[i,i+1] = -c**2/2
        M1[i+1,i] = -c**2/2

    for n in range(2,n_max+1):
        ui = 2*un - M1@un1
        unp1 = np.linalg.solve(M1,ui)
        U[n,1:N+1] = unp1
        un1 = un
        un = unp1
        
    return X, T, U



##################################################################################################

#%%

def tridiag(P, Q, R, k1=-1, k2=0, k3=1):
    """ Crée un ematrice tridiagonale sans les zeros en mémoire """
    N = len(Q)
    return (sps.spdiags(P,-1,N,N) + sps.spdiags(Q,0,N,N) + sps.spdiags(R,1,N,N))

#%%

def CSEE(N, tau, u0, nu, t_max):
    
    h = 1/(N + 1)
    
    c = nu*tau/(h**2)

    # Condition CFL
    if c>0.5:
        print("Attention ! Condition CFL non vérifiée la solution ne sera pas stable")
    else:
        print("Condition CFL vérifiée le schéma sera stable")


    # temps considéré
    #t_max = 5
    n_max = int(t_max/tau)
    T = np.linspace(0,n_max*tau,n_max+1)

    # espace considéré
    X = np.linspace(0,1,N+2)

    # Matrice des résultats
    U = np.zeros((n_max+1,N+2))
    U[0,:] = u0(X)

    # condition initiale
    u = U[0,1:N+1]

    """ # Matrice classique
    # définition de Me
    Me = np.zeros((N,N))
    for i in range(N):
        Me[i,i] = 1+2*c
    for i in range(N-1):
        Me[i,i+1] = -c
        Me[i+1,i] = -c
    """

    # définition de Me
    P = c*np.ones(N)
    Q = (1-2*c)*np.ones(N)
    R = c*np.ones(N)
  
    Me = sps.csc_matrix(tridiag(P,Q,R))

    for n in range(n_max):
        u = Me@u
        U[n+1,1:N+1]=u

    return X, T, U


#%%

def CSEI(N, tau, u0, nu, t_max):
    
    h = 1/(N + 1)
    nu = 0.1
    c = nu*tau/(h**2)

    # temps considéré
    #t_max = 5
    n_max = int(t_max/tau)
    T = np.linspace(0,n_max*tau,n_max+1)

    # espace considéré
    X = np.linspace(0,1,N+2)

    # Matrice des résultats
    U = np.zeros((n_max+1,N+2))
    U[0,:] = u0(X)

    # condition initiale
    u = U[0,1:N+1]
    """
    # définition de Me
    P = -c*np.ones(N)
    Q = (1+2*c)*np.ones(N)
    R = -c*np.ones(N)
    Mi = sps.csc_matrix(tridiag(P,Q,R))
    """
    
    # définition de Mi
    Mi = np.zeros((N,N))
    for i in range(N):
        Mi[i,i] = 1+2*c
    for i in range(N-1):
        Mi[i,i+1] = -c
        Mi[i+1,i] = -c
    
    

    for n in range(n_max):
        u = np.linalg.solve(Mi,u)
        U[n+1,1:N+1]=u

    return X, T, U

#%%

def CSCN(N, tau, u0, nu, t_max):
    
    h = 1/(N + 1)
    nu = 0.1
    c = nu*tau/(h**2)

    # temps considéré
    #t_max = 5
    n_max = int(t_max/tau)
    T = np.linspace(0,n_max*tau,n_max+1)

    # espace considéré
    X = np.linspace(0,1,N+2)

    # Matrice des résultats
    U = np.zeros((n_max+1,N+2))
    U[0,:] = u0(X)

    # condition initiale
    u = U[0,1:N+1]

    
    # définition de M1
    M1 = np.zeros((N,N))
    for i in range(N):
        M1[i,i] = 1+2*c
    for i in range(N-1):
        M1[i,i+1] = -c/2
        M1[i+1,i] = -c/2
    
    # définition de M2
    M2 = np.zeros((N,N))
    for i in range(N):
        M2[i,i] = 1-c
    for i in range(N-1):
        M2[i,i+1] = -c/2
        M2[i+1,i] = -c/2
    

    for n in range(n_max):
        ui = M2@u
        u = np.linalg.solve(M1,ui)
        U[n+1,1:N+1]=u

    return X, T, U


##################################################################################################

#%%

import numpy as np

def ASEC(N, tau, u0, V, t_max):
    
    h = 1/N
    #V = 3
    c = (V*tau)/h

    # temps considéré
    #t_max = 1
    n_max = int(t_max/tau)
    T = np.linspace(0,n_max*tau,n_max+1)
    #print(len(T))

    # espace considéré
    X = np.linspace(0,1,N+1)
    #print(len(X))

    # Matrice des résultats
    U = np.zeros((n_max+1,N+1))
    U[0,:] = u0(X)

    # condition initiale
    u = U[0,1:N+1]
    #print(U.shape)
    
    Mi = np.zeros((N,N))
    for i in range(N):
        Mi[i,i] = 1+2*c
    for i in range(N-1):
        Mi[i, i+1] = -c
        Mi[i+1, i] = -c

    for n in range(n_max):
        u = Mi@u
        U[n,0:N] = u

    return X, T, U



#%%

def ASIC(N, tau, u0, V, t_max):
    
    h = 1/N
    #V = 3
    c = (V*tau)/h


    # temps considéré
    #t_max = 1
    n_max = int(t_max/tau)
    T = np.linspace(0,n_max*tau,n_max+1)
    #print(len(T))

    # espace considéré
    X = np.linspace(0,1,N+1)
    #print(len(X))

    # Matrice des résultats
    U = np.zeros((n_max+1,N+1))
    U[0,:] = u0(X)

    # condition initiale
    u = U[0,1:N+1]
    #print(U.shape)
    
    # définition de Mi
    Mi = np.zeros((N,N))
    for i in range(N):
        Mi[i,i] = 1
    for i in range(N-1):
        Mi[i,i+1] = c/2
        Mi[i+1,i] = -c/2
    Mi[N-1,0], Mi[0,N-1] = c/2, -c/2

    #print(Mi.shape)
    #print("u", u.shape)
    for n in range(n_max):
        u = np.linalg.solve(Mi,u)
        U[n,0:N]=u

    return X, T, U

#%%


def ASEDA(N, tau, u0, V, t_max):
    
    h = 1/N
    #V = 3
    c = (V*tau)/h

    # Condition CFL
    if c>1:
        print("Attention ! Condition CFL non vérifiée la solution ne sera pas stable")
    else:
        print("Condition CFL vérifiée le schéma sera stable")



    # temps considéré
    #t_max = 1
    n_max = int(t_max/tau)
    T = np.linspace(0,n_max*tau,n_max+1)
    #print(len(T))

    # espace considéré
    X = np.linspace(0,1,N+1)
    #print(len(X))

    # Matrice des résultats
    U = np.zeros((n_max+1,N+1))
    U[0,:] = u0(X)

    # condition initiale
    u = U[0,1:N+1]
    #print(U.shape)
    
    # définition de Mi
    Mi = np.zeros((N,N))
    for i in range(N):
        Mi[i,i] = 1-c
    for i in range(N-1):
        Mi[i+1,i] = c
    Mi[0,N-1] = c

    #print(Mi.shape)
    #print("u", u.shape)
    for n in range(n_max):
        u = Mi@u
        U[n,1:N+1]=u

    return X, T, U

#%%

def ASLF(N, tau, u0, V, t_max):
    
    h = 1/N
    #V = 3
    c = (V*tau)/h

    # Condition CFL
    if c>1:
        print("Attention ! Condition CFL non vérifiée la solution ne sera pas stable")
    else:
        print("Condition CFL vérifiée le schéma sera stable")



    # temps considéré
    #t_max = 1
    n_max = int(t_max/tau)
    T = np.linspace(0,n_max*tau,n_max+1)
    #print(len(T))

    # espace considéré
    X = np.linspace(0,1,N+1)
    #print(len(X))

    # Matrice des résultats
    U = np.zeros((n_max+1,N+1))
    U[0,:] = u0(X)

    # condition initiale
    u = U[0,1:N+1]
    #print(U.shape)
    
    #Définition de la Matrice M
    M = np.zeros((N,N))

    for i in range(N-1):
        M[i,i+1] = 1/2 - c/2
        M[i+1,i] = 1/2 + c/2
    
    M[0,N-1] = 1/2 + c/2
    M[N-1,0] = 1/2 - c/2

    #Calcule d'itération
    for n in range(n_max):
        u = M@u
        U[n,0:N] = u

    return X, T, U

#%%

def ASLW(N, tau, u0, V, t_max):
    
    h = 1/N
    #V = 3
    c = (V*tau)/h

    # Condition CFL
    if c>1:
        print("Attention ! Condition CFL non vérifiée la solution ne sera pas stable")
    else:
        print("Condition CFL vérifiée le schéma sera stable")

    # temps considéré
    #t_max = 1
    n_max = int(t_max/tau)
    T = np.linspace(0,n_max*tau,n_max+1)
    #print(len(T))

    # espace considéré
    X = np.linspace(0,1,N+1)
    #print(len(X))

    # Matrice des résultats
    U = np.zeros((n_max+1,N+1))
    U[0,:] = u0(X)

    # condition initiale
    u = U[0,1:N+1]
    #print(U.shape)
    
    #Définition de la Matrice M
    M = np.zeros((N,N))

    for i in range(N):
        M[i,i] = 1-c**2

    for i in range(N-1):
        M[i,i+1] = (c**2 - c)/2
        M[i+1,i] = (c**2 + c)/2
    
    M[0,N-1] = (c**2 + c)/2
    M[N-1,0] = (c**2 - c)/2


    #Calcule d'itération
    for n in range(n_max):
        u = M@u
        U[n,0:N] = u

    return X, T, U