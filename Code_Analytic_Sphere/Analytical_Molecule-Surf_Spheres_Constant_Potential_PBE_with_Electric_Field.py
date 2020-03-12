import numpy
from numpy import pi
from scipy import special, linalg
from scipy.misc import factorial
from math import gamma

def molecule_constant_potential(q, xq, phi02, Efield, r1, r2, R, kappa, E_1, E_2):
    """
    It computes the interaction energy between a molecule (sphere with
    point-charge in the center) and a sphere at constant potential, immersed
    in water.
    Arguments
    ----------
    q      : float, number of qe to be asigned to the charge.
    phi02  : float, constant potential on the surface of the sphere 2.
    r1     : float, radius of sphere 1, i.e the molecule.
    r2     : float, radius of sphere 2.
    R      : float, distance center to center.
    kappa  : float, reciprocal of Debye length.
    E_1    : float, dielectric constant inside the sphere/molecule.
    E_2    : float, dielectric constant outside the sphere/molecule.
    Returns
    --------
    E_inter: float, interaction energy.
    """
    
    N = 20  # Number of terms in expansion

    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184

    index2 = numpy.arange(N + 1, dtype=float) + 0.5
    index = index2[0:-1]

    K1 = special.kv(index2, kappa * r1)
    K1p = index / (kappa * r1) * K1[0:-1] - K1[1:]
    k1 = special.kv(index, kappa * r1) * numpy.sqrt(pi / (2 * kappa * r1))
    k1p = -numpy.sqrt(pi / 2) * 1 / (2 * (kappa * r1)**(3 / 2.)) * special.kv(
        index, kappa * r1) + numpy.sqrt(pi / (2 * kappa * r1)) * K1p

    K2 = special.kv(index2, kappa * r2)
    K2p = index / (kappa * r2) * K2[0:-1] - K2[1:]
    k2 = special.kv(index, kappa * r2) * numpy.sqrt(pi / (2 * kappa * r2))
    k2p = -numpy.sqrt(pi / 2) * 1 / (2 * (kappa * r2)**(3 / 2.)) * special.kv(
        index, kappa * r2) + numpy.sqrt(pi / (2 * kappa * r2)) * K2p

    I1 = special.iv(index2, kappa * r1)
    I1p = index / (kappa * r1) * I1[0:-1] + I1[1:]
    i1 = special.iv(index, kappa * r1) * numpy.sqrt(pi / (2 * kappa * r1))
    i1p = -numpy.sqrt(pi / 2) * 1 / (2 * (kappa * r1)**(3 / 2.)) * special.iv(
        index, kappa * r1) + numpy.sqrt(pi / (2 * kappa * r1)) * I1p

    I2 = special.iv(index2, kappa * r2)
    I2p = index / (kappa * r2) * I2[0:-1] + I2[1:]
    i2 = special.iv(index, kappa * r2) * numpy.sqrt(pi / (2 * kappa * r2))
    i2p = -numpy.sqrt(pi / 2) * 1 / (2 * (kappa * r2)**(3 / 2.)) * special.iv(
        index, kappa * r2) + numpy.sqrt(pi / (2 * kappa * r2)) * I2p

    B = numpy.zeros((N, N), dtype=float)
    for n in range(N):
        for m in range(N):
            for nu in range(N):
                if n >= nu and m >= nu:
                    g1 = gamma(n - nu + 0.5)
                    g2 = gamma(m - nu + 0.5)
                    g3 = gamma(nu + 0.5)
                    g4 = gamma(m + n - nu + 1.5)
                    f1 = factorial(n + m - nu)
                    f2 = factorial(n - nu)
                    f3 = factorial(m - nu)
                    f4 = factorial(nu)
                    Anm = g1 * g2 * g3 * f1 * (n + m - 2 * nu + 0.5) / (
                        pi * g4 * f2 * f3 * f4)
                    kB = special.kv(n + m - 2 * nu + 0.5, kappa *
                                    R) * numpy.sqrt(pi / (2 * kappa * R))
                    B[n, m] += Anm * kB

    E_hat = E_1 / E_2
    M = numpy.zeros((2 * N, 2 * N), float)
    for j in range(N):
        for n in range(N):
            M[j, n + N] = (2 * j + 1) * B[j, n] * (
                kappa * i1p[j] / k2[n] - E_hat * j / r1 * i1[j] / k2[n])
            M[j + N, n] = (2 * j + 1) * B[j, n] * i2[j] * 1 / (
                kappa * k1p[n] - E_hat * n / r1 * k1[n])
            if n == j:
                M[j, n] = 1
                M[j + N, n + N] = 1


    RHS = numpy.zeros(2 * N)
    rho = numpy.sqrt(sum(xq[0]**2)) ##
    zenit = numpy.arccos((xq[0,2])/rho) ##

    for ss in range(N):
        RHS[ss] = -E_hat*((q)/(4*pi*E_1))*((rho**ss)/(r1**(ss+2)))*(2*ss + 1) 
        if ss == 1:
            RHS[1] = -E_hat*((q)/(4*pi*E_1))*((rho**ss)/(r1**(ss+2)))*(2*ss + 1) - Efield*(E_hat*ss - 1)
        
    RHS[N] = phi02 + Efield*R
    RHS[N+1] = Efield*r2
    coeff = linalg.solve(M, RHS)

    a = coeff[0:N] / (kappa * k1p - E_hat * numpy.arange(N) / r1 * k1)
    b = coeff[N:2 * N] / k2

    Cn = numpy.zeros(N)
    for jj in range(len(Cn)):
        Cn[jj] = (a[jj] * k1[jj]  + i1[jj] * (2*jj + 1) * numpy.sum(b * B[:,jj]) - (((q)/(4*pi*E_1))*((rho**jj)/(r1**(jj + 1)))))/(r1**jj)
        if jj == 1:
            Cn[jj] = (a[jj] * k1[jj]  + i1[jj] * (2*jj + 1) * numpy.sum(b * B[:,jj]) - Efield*r1 - (((q)/(4*pi*E_1))*((rho**jj)/(r1**(jj + 1)))))/(r1**jj)
        

    a_inf = numpy.zeros(N)
    for k in range(N):
        a_inf[k] = - (E_hat*(q/(4*pi*E_1))*((rho**k)/(r1**(k+2)))*(2*k+1))/(kappa*k1p[k] - E_hat*(k/r1)*k1[k])
        if k == 1:
            a_inf[k] = - ((Efield)*(E_hat*k - 1) + (E_hat*(q/(4*pi*E_1))*((rho**k)/(r1**(k+2)))*(2*k+1)))/(kappa*k1p[k] - E_hat*(k/r1)*k1[k])
        

    Cn_inf = numpy.zeros(N)
    for j in range(N):
        Cn_inf[j] = (a_inf[j]*k1[j] - (((q)/(4*pi*E_1))*((rho**j)/(r1**(j + 1)))))/(r1**j)
        if j == 1:
            Cn_inf[j] = (a_inf[j]*k1[j] - Efield*r1- (((q)/(4*pi*E_1))*((rho**j)/(r1**(j + 1)))))/(r1**j)
        

    phi_solv_mol_surf = 0
    phi_solv_mol_surf_inf = 0
    for kk in range(N):
        phi_solv_mol_surf += Cn[kk]*(rho**(kk))*special.lpmv(0,kk,numpy.cos(zenit))
        phi_solv_mol_surf_inf += Cn_inf[kk]*(rho**(kk))*special.lpmv(0,kk,numpy.cos(zenit))

    Gp_comp1 = 0
    Gp_comp2 = 0
    for n in range(N):
        Gp_comp1 += b[n]*k2p[n]*special.lpmv(0,n,numpy.cos(zenit))
        for m in range(N):
            Gp_comp2 += a[n]*(2*m + 1)*B[n,m]*i2p[m]*special.lpmv(0,m,numpy.cos(zenit))

    Gp = Gp_comp1 + Gp_comp2 - Efield*special.lpmv(0,1,numpy.cos(zenit))

    C0 = qe**2 * Na * 1e-3 * 1e10 / (cal2J * E_0)
    C1 = q * 0.5
    C2 = 2 * pi * kappa * phi02 * r2 * r2 * E_2

    phi_surf_mol_surf = Gp

    Solv_Energy_h = phi_solv_mol_surf*C0*C1
    Solv_Energy_inf = phi_solv_mol_surf_inf*C0*C1
    Surf_Energy_h = phi_surf_mol_surf*C0*C2

    #Evaluar potencial en r1.. phi1 y phi3 deben ser iguales en la interfaz 1
    #charge_pot = 0
    #phi_interfaz1 = 0
    #for j in range(N):
        #charge_pot += (q/(4*pi*E_1))*((rho**j)/(r1**(j+1)))*special.lpmv(0,j,numpy.cos(zenit))
        #phi_interfaz1 += Cn[j]*(r1**(j))*special.lpmv(0,j,numpy.cos(zenit))

    #phi1_interfaz1 = charge_pot + phi_interfaz1
    #print('phi1_interfaz_1: ', phi1_interfaz1)

    #componente1_phi3 = 0
    #componente2_phi3 = 0
    #for n in range(N):
        #componente1_phi3 += a[n]*k1[n]*special.lpmv(0,n,numpy.cos(zenit))
        #for m in range(N):
            #componente2_phi3 += b[n]*(2*m + 1)*B[n,m]*i1[m]*special.lpmv(0,m,numpy.cos(zenit))

    #phi3_interfaz1 = componente1_phi3 + componente2_phi3
    #print('phi3_interfaz_1: ', phi3_interfaz1)


    return Solv_Energy_h, Solv_Energy_inf, Surf_Energy_h


q= 1
xq = numpy.array([[1e-12,1e-12,0.5]])
#xq = numpy.array([[1e-12,1e-12,1e-12]])
phi02 = 1.
Efield = 1.1053e-3
r1 = 1.
r2 = 1.
R = 12.
#kappa = 0.0000001
kappa = 0.125
E_1 = 4.
E_2 = 80.


Energy_Mol_Surf_Cte_Potential = molecule_constant_potential(q, xq, phi02, Efield, r1, r2, R, kappa, E_1, E_2)
print("========================================================================")
print('Mol - Surf Cte. Potential -- Energy: ')
print('Esolv Mol-Surf: ', Energy_Mol_Surf_Cte_Potential[0], ' kcal/mol')
print('Esolv Mol Inf: ', Energy_Mol_Surf_Cte_Potential[1], ' kcal/mol')
print('Esurf Mol-Surf: ', Energy_Mol_Surf_Cte_Potential[2], ' kcal/mol')
print("========================================================================")
