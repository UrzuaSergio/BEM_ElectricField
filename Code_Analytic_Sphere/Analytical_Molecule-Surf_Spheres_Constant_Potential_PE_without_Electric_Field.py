# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 22:29:05 2020

@author: Sergio
"""

import numpy
from numpy import pi
from scipy import special, linalg
from scipy.misc import factorial
from math import gamma

def molecule_constant_potential_laplace_centered_charge(q, xq, phi02, r1, r2, R, E_1, E_2):
    
    Param = 0.
    N = 20   
    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184
    
    E_hat = E_1 / E_2
    
    for K in range(len(q)):
        rho = numpy.sqrt(sum(xq[K]**2))
        zenit = numpy.arccos(xq[K,2]/rho)
        azim  = numpy.arctan2(xq[K,1],xq[K,0])
        
    M = numpy.zeros((2 * N, 2 * N), float)
    for j in range(N):
        for n in range(N):
            M[j, n + N] = ((r2/R)**(n + 1))*((-1)**n)*((factorial(j + n))/((factorial(j))*(factorial(n))))*((r1/R)**j)*(j/r1)*(E_hat - 1)
            M[j + N, n] = (((r1/R)**(n + 1))*((factorial(j + n))/((factorial(j))*(factorial(n))))*((-1)**j)*((r2/R)**j))/((1/r1)*(n*E_hat + n + 1)) 
            if n == j:
                M[j, n] = 1
                M[j + N, n + N] = 1

    RHS = numpy.zeros(2*N)
    RHS[0] = E_hat * q[0] / (4 * pi * E_1 * r1 * r1)
    RHS[N] = phi02
    coeff = linalg.solve(M, RHS)
    
    an = coeff[0:N] / ((1/r1)*((numpy.arange(N))*(E_hat + 1) + 1))
    An = coeff[N:2 * N]

    a0 = an[0]
    A0 = An[0]
    
    SUMA = 0
    for n in range(N):
        SUMA += An[n]*((r2/R)**(n + 1))*((-1)**n)
     
    
    phi_h = a0 + SUMA - q[0] / (4 * pi * E_1 * r1)
    
    C0 = qe**2 * Na * 1e-3 * 1e10 / (cal2J * E_0)
    C1 = q[0] * 0.5
    
    E_solv_mol_surf = phi_h*C0*C1

    E_surf_mol_surf = - 2*pi*E_2*A0*phi02*r2*C0
        
    return E_solv_mol_surf, E_surf_mol_surf

def molecule_constant_potential_laplace_move_charge(q, xq, phi02, r1, r2, R, E_1, E_2):
    
    Param = 0.
    N = 20   
    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184
    
    E_hat = E_1 / E_2
    
    for K in range(len(q)):
        rho = numpy.sqrt(sum(xq[K]**2))
        zenit = numpy.arccos(xq[K,2]/rho)
        azim  = numpy.arctan2(xq[K,1],xq[K,0])
    
         
    M = numpy.zeros((2 * N, 2 * N), float)
    for j in range(N):
        for n in range(N):
            M[j, n + N] = ((r2/R)**(n + 1))*((-1)**n)*((factorial(j + n))/((factorial(j))*(factorial(n))))*((r1/R)**j)*(j/r1)*(E_hat - 1)
            M[j + N, n] = (((r1/R)**(n + 1))*((factorial(j + n))/((factorial(j))*(factorial(n))))*((-1)**j)*((r2/R)**j))/((1/r1)*(n*E_hat + n + 1)) 
            if n == j:
                M[j, n] = 1
                M[j + N, n + N] = 1
                
    RHS = numpy.zeros(2*N)
    for j in range(N):
        RHS[j] = E_hat*((q[0])/(4*pi*E_1))*(((rho**j))/(r1**(j+2)))*(2*j + 1)         
    
    RHS[N] = phi02
    coeff = linalg.solve(M, RHS)
    
    an = coeff[0:N] / ((1/r1)*((numpy.arange(N))*(E_hat + 1) + 1))
    An = coeff[N:2 * N]
    
    Cj = numpy.zeros(N)
    for j in range(N):
        SUMA = 0
        for n in range(N):
            SUMA += An[n]*((r2/R)**(n + 1))*((-1)**n)*((factorial(j + n))/((factorial(j))*(factorial(n))))*((r1/R)**j)    
       
        Cj[j] = an[j] + SUMA - ((q[0])/(4*pi*E_1))*(((rho**j))/(r1**(j + 1)))

    phi_solv_mol_surf = 0
    for n in range(N):
        phi_solv_mol_surf += Cj[n]*((rho/r1)**n)*special.lpmv(0,n,numpy.cos(zenit))
        
    C0 = qe**2 * Na * 1e-3 * 1e10 / (cal2J * E_0)
    C1 = q[0] * 0.5
    
    E_solv_mol_surf = phi_solv_mol_surf*C0*C1
    
    Gp_aux1 = 0.
    Gp_aux2 = 0.
    for n in range(N):
        Gp_aux1 += - An[n]*((n + 1)/r2)*special.lpmv(0,n,numpy.cos(zenit))
        for m in range(N):
            Gp_aux2 += an[n]*((r1/R)**(n + 1))*((factorial(m + n))/((factorial(m))*(factorial(n))))*((-1)**m)*(m/r2)*((r2/R)**m)*special.lpmv(0,m,numpy.cos(zenit))
            
            
    Gp = (Gp_aux1 + Gp_aux2)*(-1)*(E_2/phi02)

    E_surf_mol_surf = -2*pi*(r2**2)*(phi02**2)*Gp*C0
               
    return E_solv_mol_surf, E_surf_mol_surf

q   = numpy.array([1.])
xq  = numpy.array([[1e-12,1e-12,1e-12]])
xq_move = numpy.array([[1e-12,1e-12,2.0]])
phi02 = 1.
r1 = 4.
r2 = 4.
R = 12.
E_1 = 4.
E_2 = 80.

Energia = molecule_constant_potential_laplace_centered_charge(q, xq, phi02, r1, r2, R, E_1, E_2)
print("===============================================")
print("Caso: 'centered charge': ")
print("E_solv_mol-surf: ", Energia[0], "[kcal/mol]")
print("E_surf_mol-surf: ", Energia[1], "[kcal/mol]")
print("===============================================")
Energia2 = molecule_constant_potential_laplace_move_charge(q, xq_move, phi02, r1, r2, R, E_1, E_2)
print("Caso: 'off center charge': ")
print("E_solv_mol-surf: ", Energia2[0], "[kcal/mol]")
print("E_surf_mol-surf: ", Energia2[1], "[kcal/mol]")
