# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 17:02:54 2019

@author: Cooper
"""

import numpy
from numpy import pi
from scipy import special, linalg
from scipy.misc import factorial
from math import gamma

def two_sphere(a, R, kappa, E_1, E_2, q):
    """
    It computes the analytical solution of a spherical surface and a spherical
    molecule with a center charge, both of radius R.
    Follows Cooper&Barba 2016
    Arguments
    ----------
    a    : float, center to center distance.
    R    : float, radius of surface and molecule.
    kappa: float, reciprocal of Debye length.
    E_1  : float, dielectric constant inside the sphere.
    E_2  : float, dielectric constant outside the sphere.
    q    : float, number of qe to be asigned to the charge.
    Returns
    --------
    Einter  : float, interaction energy.
    E1sphere: float, solvation energy of one sphere.
    E2sphere: float, solvation energy of two spheres together.
    Note:
    Einter should match (E2sphere - 2xE1sphere)
    """

    N = 20  # Number of terms in expansion.

    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184

    index2 = numpy.arange(N + 1, dtype=float) + 0.5
    index = index2[0:-1]

    K1 = special.kv(index2, kappa * a)
    K1p = index / (kappa * a) * K1[0:-1] - K1[1:]

    k1 = special.kv(index, kappa * a) * numpy.sqrt(pi / (2 * kappa * a))
    k1p = -numpy.sqrt(pi / 2) * 1 / (2 * (kappa * a)**(3 / 2.)) * special.kv(
        index, kappa * a) + numpy.sqrt(pi / (2 * kappa * a)) * K1p

    I1 = special.iv(index2, kappa * a)
    I1p = index / (kappa * a) * I1[0:-1] + I1[1:]
    i1 = special.iv(index, kappa * a) * numpy.sqrt(pi / (2 * kappa * a))
    i1p = -numpy.sqrt(pi / 2) * 1 / (2 * (kappa * a)**(3 / 2.)) * special.iv(
        index, kappa * a) + numpy.sqrt(pi / (2 * kappa * a)) * I1p

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

    M = numpy.zeros((N, N), float)
    E_hat = E_1 / E_2
    for i in range(N):
        for j in range(N):
            M[i, j] = (2 * i + 1) * B[i, j] * (
                kappa * i1p[i] - E_hat * i * i1[i] / a)
            if i == j:
                M[i, j] += kappa * k1p[i] - E_hat * i * k1[i] / a

    RHS = numpy.zeros(N)
    RHS[0] = -E_hat * q / (4 * pi * E_1 * a * a)

    a_coeff = linalg.solve(M, RHS)

    a0 = a_coeff[0]
    a0_inf = -E_hat * q / (4 * pi * E_1 * a * a) * 1 / (kappa * k1p[0])

    phi_2 = a0 * k1[0] + i1[0] * numpy.sum(a_coeff * B[:, 0]) - q / (4 * pi *
                                                                     E_1 * a)
    phi_1 = a0_inf * k1[0] - q / (4 * pi * E_1 * a)
    phi_inter = phi_2 - phi_1

    CC0 = qe**2 * Na * 1e-3 * 1e10 / (cal2J * E_0)

    Einter = 0.5 * CC0 * q * phi_inter
    E1sphere = 0.5 * CC0 * q * phi_1
    E2sphere = 0.5 * CC0 * q * phi_2

    return Einter, E1sphere, E2sphere


a = 4.
R = 4.
#kappa = 0.0000001
kappa = 0.125
E_1 = 4.
E_2 = 80.
q = 1
Energia = two_sphere(a, R, kappa, E_1, E_2, q)
Esolv_1sphere = Energia[1]
print(Energia[1]) 