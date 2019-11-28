import bempp.api
import numpy as np

#Importacion de malla msh

grid = bempp.api.import_grid('sphere_r4_gmsh.msh')

q, xq = np.array([]), np.empty((0,3))

Ef = 0.
cte_dielec_in = 4.
cte_dielec_ext = 80.
k = 0.125

pqr_file = open('centered_charge.pqr','r').read().split('\n')
for line in pqr_file:
    line=line.split()
    if len(line)==0 or line[0]!='ATOM': continue
    q = np.append(q,float(line[8]))
    xq = np.vstack((xq, np.array(line[5:8]).astype(float)))

def charges_fun(x, n, domain_index, result):
    global q, xq, cte_dielec_in
    result[:] = np.sum(q/np.linalg.norm(x - xq, axis=1))/(4*np.pi*cte_dielec_in)

def zero(x, n, domain_index, result):
    result[:] = 0

#Campo Electrico Externo Uniforme direccionado en el eje Z

def Efield_Pot(x, n, domain_index, result):
    global Ef
    result[:] = - Ef*x[2]

def Efield_DerPot(x, n, domain_index, result):
    global Ef
    result[:] = - Ef*n[2]

dirichl_space = bempp.api.function_space(grid, "DP", 0)
neumann_space = bempp.api.function_space(grid, "DP", 0)

charged_grid_fun = bempp.api.GridFunction(dirichl_space, fun=charges_fun)
Efield_Pot_grid_fun = bempp.api.GridFunction(neumann_space, fun=Efield_Pot)
Efield_DerPot_grid_fun = bempp.api.GridFunction(neumann_space, fun=Efield_DerPot)

from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz
identity = sparse.identity(dirichl_space, dirichl_space, dirichl_space)
slp_L_in = laplace.single_layer(neumann_space, dirichl_space, dirichl_space)
dlp_L_in = laplace.double_layer(dirichl_space, dirichl_space, dirichl_space)
slp_L_out = laplace.single_layer(neumann_space, dirichl_space, dirichl_space)
dlp_L_out = laplace.double_layer(dirichl_space, dirichl_space, dirichl_space)

rhsEf_out = (0.5*identity - dlp_L_out)*Efield_Pot_grid_fun + slp_L_out*Efield_DerPot_grid_fun

blocked = bempp.api.BlockedOperator(2, 2)
blocked[0, 0] = 0.5*identity + dlp_L_in
blocked[0, 1] = -slp_L_in
blocked[1, 0] = 0.5*identity - dlp_L_out
blocked[1, 1] = (cte_dielec_in/cte_dielec_ext)*slp_L_out

sol, info, it_count = bempp.api.linalg.gmres(blocked, [charged_grid_fun, rhsEf_out], use_strong_form=True, return_iteration_count=True, tol=1e-5)
print("El sistema lineal fue resuelto en {0} iteraciones".format(it_count))
solution_dirichl, solution_neumann = sol

#Calculo de Energia de Solvatacion

slp_q = bempp.api.operators.potential.laplace.single_layer(neumann_space, xq.transpose())
dlp_q = bempp.api.operators.potential.laplace.double_layer(dirichl_space, xq.transpose())
phi_q = slp_q*solution_neumann - dlp_q*solution_dirichl

Total_energy = 2*np.pi*332.064*np.sum(q*phi_q).real
print("Esolv: {:7.4f} [kcal/mol]".format(Total_energy))
