from numpy import *
from scipy import special
from scipy.special import lpmv
from scipy.misc import factorial
import math

# K: modified spherical Bessel function
def get_K(x,n):

    K = 0.
    n_fact = factorial(n)
    n_fact2 = factorial(2*n)
    for s in range(n+1):
        K += 2**s*n_fact*factorial(2*n-s)/(factorial(s)*n_fact2*factorial(n-s)) * x**s

    return K

# Solucion Analitica de Esfera con carga puntual en el interior.
def solution(q, xq, E_1, E_2, R, kappa, a, N):

    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184

    PHI = zeros(len(q))
    for K in range(len(q)):
        rho = sqrt(sum(xq[K]**2))
        zenit = arccos(xq[K,2]/rho)
        azim  = arctan2(xq[K,1],xq[K,0])
    
        phi = 0.+0.*1j
        for n in range(N):
            for m in range(-n,n+1):
                
                P1 = lpmv(abs(m),n,cos(zenit))

                Enm = 0.
                for k in range(len(q)):
                    rho_k   = sqrt(sum(xq[k]**2))
                    zenit_k = arccos(xq[k,2]/rho_k)
                    azim_k  = arctan2(xq[k,1],xq[k,0])
                    P2 = lpmv(abs(m),n,cos(zenit_k))

                    Enm += q[k]*rho_k**n*conjugate(special.sph_harm(m, n, azim_k, zenit_k))
#               
                C2 = (kappa*a)**2*get_K(kappa*a,n-1)/(get_K(kappa*a,n+1) +
                        n*(E_2-E_1)/((n+1)*E_2+n*E_1)*(R/a)**(2*n+1)*(kappa*a)**2*get_K(kappa*a,n-1)/((2*n-1)*(2*n+1)))
                C1 = Enm/(E_2*E_0*a**(2*n+1)) * (2*n+1)/(2*n-1) * (E_2/((n+1)*E_2+n*E_1))**2
                
                if n==0 and m==0:
                    Bnm = (real(Enm)/(E_0*R)*(1/E_2-1/E_1) - real(Enm)*kappa*a/(E_0*E_2*a*(1+kappa*a)))*4*pi/(2*n+1)
                else:
                    Bnm = (1./(E_1*E_0*R**(2*n+1)) * (E_1-E_2)*(n+1)/(E_1*n+E_2*(n+1)) * real(Enm)  - C1*C2)* 4*pi/(2*n+1)

                phi += real(Bnm)*rho**n*special.sph_harm(m, n, azim, zenit)

        PHI[K] = real(phi)/(4*pi)
        
    C0 = qe**2*Na*1e-3*1e10/(cal2J)
    CC0 = 1.
    E_P = 0.5*C0*sum(q*PHI)
    
    return E_P

q   = array([1.])
#xq  = array([[0.,0.,2.]])
xq  = array([[1e-12,1e-12,1e-12]])
E_1 = 4.
E_2 = 80.
R   = 4.
kappa = 1e-12
a   = 1.
N   = 20

print('================================================================')
print("Solución Analitica: esfera con una carga puntual en el centro.")
print('================================================================')
print("Esolv: ", solution(q, xq, E_1, E_2, R, kappa, a, N)," [kcal/mol]")
print('================================================================')