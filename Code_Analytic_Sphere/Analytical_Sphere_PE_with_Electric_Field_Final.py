from numpy import *
from scipy import special
from scipy.special import lpmv, legendre
from scipy.misc import factorial

def sphere_with_Electric_Field(q, xq, R, Ep_1, Ep_2, EF, N):
    """
    =====================================================================
    Inputs:
    =====================================================================
    q    :  Array, Valor de carga electrica. (codigo para una sola carga)
    xq   :  Array, Posición de Carga.
    R    :  Float, Radio de la esfera.
    Ep1  :  Float, Constante Dielectrica interior de esfera.
    Ep2  :  Float, Constante Dielectrica exterior de esfera.
    EF   :  Float, Valor de Campo Electrico.
    N    :  Int, Numero de terminos en la Expansion Esferica Armonica.
    =====================================================================
    Output
    =====================================================================
    Esolv: Float, Energia de Solvatacion.
    =====================================================================
    """

    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184

    E_hat = Ep_1/Ep_2

    for K in range(len(q)):
        rho = sqrt(sum(xq[K]**2))
        zenit = arccos(xq[K,2]/rho)
        azim  = arctan2(xq[K,1],xq[K,0])

    print('rho: ', rho)
    print('zenit: ', zenit)
    Matrix = zeros((2 * (N+1), 2 * (N+1)), float)
    for j in range(N+1):
        for n in range(N+1):
            Matrix[j, j + N+1] = - R**(-(j + 1)) #Cj de eq. phi1 = phi2
            Matrix[j + N+1, j] = E_hat*j*(R**(j-1)) #Aj de eq. Ep_1*dphi1/dn = Ep_2*dphi2/dn
            if n == j:
                Matrix[j, n] = R**(j) #Aj de eq. phi1 = phi2
                Matrix[j + N+1, n + N+1] = (j+1)*(R**(-(j+2))) #Cj de eq. Ep_1*dphi1/dn = Ep_2*dphi2/dn

    RHS = zeros(2 * (N+1))
    print('========================================================')
    print("Matriz: ")
    print(Matrix)

    for i in range(N+1):
        RHS[i] = -(q[0]/(4*pi*Ep_1))*((rho**i)/(R**(i+1)))
        RHS[N+1+i] = (q[0]/(4*pi*Ep_2))*((rho**i)/(R**(i+2)))*(i+1)
        if i == 1:
            RHS[i] = -EF*R -(q[0]/(4*pi*Ep_1))*((rho**i)/(R**(i+1)))
            RHS[N+1+i] = -EF + (q[0]/(4*pi*Ep_2))*((rho**i)/(R**(i+2)))*(i+1)

    print('========================================================')
    print('Lado Derecho: ')
    print(RHS)
    print('========================================================')
    coeff = linalg.solve(Matrix, RHS)
    print("Coeficientes [Aj,Cj]: ")
    print(coeff)

    suma = 0

    print('========================================================')

    suma_reac = 0.
    for s in range(N+1):
        suma += coeff[s]*(R**s)*lpmv(0,s,cos(zenit))#Pn[s]
        suma_reac += coeff[s]*(rho**s)*lpmv(0,s,cos(zenit))#Pn[0]


    print('========================================================')
    print("Solución Analitica: esfera con una carga electrica ")
    print("puntual, sometida a un campo electrico uniforme externo.")
    
    print('========================================================')
    print('========================================================')
    print('Potencial en la Interfaz (r = R)')
    PHI1 = suma + (q[0]/(4*pi*Ep_1*R))
    print('phi1(R): ', PHI1)
#    print('========================================================')

    suma1 = 0
    for c in range(N+1):
        suma1 += coeff[c+N+1]*(R**(-(c+1)))*lpmv(0,c,cos(zenit))


    PHI2 = suma1 - EF*R*lpmv(0,1,cos(zenit))
    print('phi2(R): ', PHI2)

    print('========================================================')
    print('Potencial de reacción:')
    Phi_total = suma_reac + (q[0]/(4*pi*Ep_1*R))
    Phi_coul = (q[0]/(4*pi*Ep_1*R))
    Phi_reac = Phi_total - Phi_coul
    print('Phi_reac: ', Phi_reac)

    CC0 = qe**2 * Na * 1e-3 * 1e10 / (cal2J * E_0)
    E_solv = 0.5*CC0*sum(q[0]*Phi_reac)

    return E_solv

N = 20
q   = array([1.])
xq  = array([[0.,0.,2.]])
#xq  = array([[1e-12,1e-12,1e-12]])
R = 4.
Ep_1 = 4.
Ep_2 = 80.
EF = 2.4

E_solv = sphere_with_Electric_Field(q, xq, R, Ep_1, Ep_2, EF, N)
print('========================================================')
print('Esolv: ', E_solv, ' kcal/mol')
print('========================================================')
