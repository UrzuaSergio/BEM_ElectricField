import numpy
from numpy import pi
from scipy import special, linalg
from scipy.misc import factorial
from math import gamma

def molecule_constant_charge(q, sigma02, r1, r2, R, kappa, E_1, E_2):
    """
    It computes the interaction energy between a molecule (sphere with
    point-charge in the center) and a sphere at constant charge, immersed
    in water.

    Arguments
    ----------
    q      : float, number of qe to be asigned to the charge.
    sigma02: float, constant charge on the surface of the sphere 2.
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
                i1p[j] / k2p[n] - E_hat * j / r1 * i1[j] / (kappa * k2p[n]))
            M[j + N, n] = (2 * j + 1) * B[j, n] * i2p[j] * kappa * 1 / (
                kappa * k1p[n] - E_hat * n / r1 * k1[n])
            if n == j:
                M[j, n] = 1
                M[j + N, n + N] = 1

    RHS = numpy.zeros(2 * N)
    RHS[0] = -E_hat * q / (4 * pi * E_1 * r1 * r1)
    RHS[N] = -sigma02 / E_2

    coeff = linalg.solve(M, RHS)

    a = coeff[0:N] / (kappa * k1p - E_hat * numpy.arange(N) / r1 * k1)
    b = coeff[N:2 * N] / (kappa * k2p)

    a0 = a[0]
    a0_inf = -E_hat * q / (4 * pi * E_1 * r1 * r1) * 1 / (kappa * k1p[0])
    b0 = b[0]
    b0_inf = -sigma02 / (E_2 * kappa * k2p[0])

    phi_inf = a0_inf * k1[0] - q / (4 * pi * E_1 * r1)
    phi_h = a0 * k1[0] + i1[0] * numpy.sum(b * B[:, 0]) - q / (4 * pi * E_1 *
                                                               r1)
    phi_inter = phi_h - phi_inf

    U_inf = b0_inf * k2[0]
    U_h = b0 * k2[0] + i2[0] * numpy.sum(a * B[:, 0])
    U_inter = U_h - U_inf
    

    C0 = qe**2 * Na * 1e-3 * 1e10 / (cal2J * E_0)
    C1 = q * 0.5
    C2 = 2 * pi * sigma02 * r2 * r2 
    E_inter = C0 * (C1 * phi_inter + C2 * U_inter)
    print("Const Charge: Esolv_Mol-Surf: ", phi_h*C0*C1," - Esurf_Mol-surf: ", U_h*C0*C2)
    
    return E_inter

def molecule_constant_potential(q, phi02, r1, r2, R, kappa, E_1, E_2):
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
    RHS[0] = -E_hat * q / (4 * pi * E_1 * r1 * r1)
    RHS[N] = phi02
    print(RHS)
    coeff = linalg.solve(M, RHS)

    a = coeff[0:N] / (kappa * k1p - E_hat * numpy.arange(N) / r1 * k1)
    b = coeff[N:2 * N] / k2

    a0 = a[0]
    a0_inf = -E_hat * q / (4 * pi * E_1 * r1 * r1) * 1 / (kappa * k1p[0])
    b0 = b[0]
    b0_inf = phi02 / k2[0]

    phi_inf = a0_inf * k1[0] - q / (4 * pi * E_1 * r1)
    phi_h = a0 * k1[0] + i1[0] * numpy.sum(b * B[:, 0]) - q / (4 * pi * E_1 *
                                                               r1)
    phi_inter = phi_h - phi_inf
    

    U_inf = b0_inf * k2p[0]
    U_h = b0 * k2p[0] + i2p[0] * numpy.sum(a * B[:, 0])
    U_inter = U_h - U_inf

    C0 = qe**2 * Na * 1e-3 * 1e10 / (cal2J * E_0)
    C1 = q * 0.5
    C2 = 2 * pi * kappa * phi02 * r2 * r2 * E_2
    E_inter = C0 * (C1 * phi_inter + C2 * U_inter)
    print("Const Pot: Esolv_Mol-Surf: ",phi_h*C0*C1," -  Esurf_Mol-Surf: ", U_h*C0*C2)

    return E_inter

qe = 1.60217646e-19
E_0 = 8.854187818e-12
#Pot_el = Pot*(qe)/((1e-10)*(E_0))

q= 1
sigma02 = 80. #(0.05 C/m^2)
phi02 = 1.
r1 = 4.
r2 = 4.
R = 12.
kappa = 0.0000001
#kappa = 0.125
E_1 = 4.
E_2 = 80.

Energy_Mol_Surf_Cte_Charge = molecule_constant_charge(q, sigma02, r1, r2, R, kappa, E_1, E_2)
Energy_Mol_Surf_Cte_Potential = molecule_constant_potential(q, phi02, r1, r2, R, kappa, E_1, E_2)

print('Mol - Surf Cte. Charge -- E_int: ', Energy_Mol_Surf_Cte_Charge, ' kcal/mol')
print('Mol - Surf Cte. Potential -- E_int: ', Energy_Mol_Surf_Cte_Potential, ' kcal/mol')
