import bempp.api
import numpy as np

#Importacion de malla msh

grid1 = bempp.api.import_grid('sphere_r4_gmsh0.25.msh')
grid2 = bempp.api.import_grid('sphere_r4_d12_gmsh0.25.msh')


numero_de_elementos_malla_1 = grid1.leaf_view.entity_count(0)
numero_de_elementos_malla_2 = grid2.leaf_view.entity_count(0)
print("La malla de la molecula tiene {0} elementos.".format(numero_de_elementos_malla_1))
print("La malla de la superficie tiene {0} elementos.".format(numero_de_elementos_malla_2))


q, xq = np.array([]), np.empty((0,3))

Ef = 0.
sigma02 = 1.
cte_dielec_in = 4.
cte_dielec_ext = 80.
#k = 0.0001
k=0.125

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

#Campo Electrico Externo Uniforme direccionado en el eje coordenado Z

def Efield_Pot(x, n, domain_index, result):
    global Ef
    result[:] = - Ef*x[2]

def Efield_DerPot(x, n, domain_index, result):
    global Ef
    result[:] = - Ef*n[2]

def Sigma02(x, n, domain_index, result):
    global sigma02, cte_dielec_ext
    result[:] = sigma02/cte_dielec_ext

dirichl_space_1 = bempp.api.function_space(grid1, "DP", 0)
neumann_space_1 = bempp.api.function_space(grid1, "DP", 0)
dirichl_space_2 = bempp.api.function_space(grid2, "DP", 0)
neumann_space_2 = bempp.api.function_space(grid2, "DP", 0)

charged_grid_fun_1 = bempp.api.GridFunction(dirichl_space_1, fun=charges_fun)
Efield_Pot_grid_fun_1 = bempp.api.GridFunction(neumann_space_1, fun=Efield_Pot)
Efield_DerPot_grid_fun_1 = bempp.api.GridFunction(neumann_space_1, fun=Efield_DerPot)
Efield_Pot_grid_fun_2 = bempp.api.GridFunction(neumann_space_2, fun=Efield_Pot)
Efield_DerPot_grid_fun_2 = bempp.api.GridFunction(neumann_space_2, fun=Efield_DerPot)
Sigma02_grid_fun = bempp.api.GridFunction(neumann_space_2, fun=Sigma02)

from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz
identity_11 = sparse.identity(dirichl_space_1, dirichl_space_1, dirichl_space_1)
slp_L_11 = laplace.single_layer(neumann_space_1, dirichl_space_1, dirichl_space_1)
dlp_L_11 = laplace.double_layer(dirichl_space_1, dirichl_space_1, dirichl_space_1)

slp_Y_11 = modified_helmholtz.single_layer(neumann_space_1, dirichl_space_1, dirichl_space_1, k)
dlp_Y_11 = modified_helmholtz.double_layer(dirichl_space_1, dirichl_space_1, dirichl_space_1, k)
dlp_Y_12 = modified_helmholtz.double_layer(dirichl_space_2, dirichl_space_1, dirichl_space_1, k)
slp_Y_12 = modified_helmholtz.single_layer(neumann_space_2, dirichl_space_1, dirichl_space_1, k)

slp_Y_21 = modified_helmholtz.single_layer(neumann_space_1, dirichl_space_2, dirichl_space_2, k)
dlp_Y_21 = modified_helmholtz.double_layer(dirichl_space_1, dirichl_space_2, dirichl_space_2, k)

identity_22 = sparse.identity(dirichl_space_2, dirichl_space_2, dirichl_space_2)
slp_Y_22 = modified_helmholtz.single_layer(neumann_space_2, dirichl_space_2, dirichl_space_2, k)
dlp_Y_22 = modified_helmholtz.double_layer(dirichl_space_2, dirichl_space_2, dirichl_space_2, k)


#Lado derecho de la ecuacion de Poisson-Boltzmann Modificada con Campo Electrico

rhsEf_out_1 = slp_Y_12*(Sigma02_grid_fun)
rhsEf_out_2 = slp_Y_22*(Sigma02_grid_fun)

#matriz

blocked = bempp.api.BlockedOperator(3, 3)
blocked[0, 0] = 0.5*identity_11 + dlp_L_11
blocked[0, 1] = -slp_L_11
blocked[1, 0] = 0.5*identity_11 - dlp_Y_11
blocked[1, 1] = (cte_dielec_in/cte_dielec_ext)*slp_Y_11
blocked[1, 2] = - dlp_Y_12
blocked[2, 0] =  - dlp_Y_21
blocked[2, 1] = (cte_dielec_in/cte_dielec_ext)*slp_Y_21
blocked[2, 2] = 0.5*identity_22 -dlp_Y_22


sol, info, it_count = bempp.api.linalg.gmres(blocked, [charged_grid_fun_1, rhsEf_out_1, rhsEf_out_2], use_strong_form=True, return_iteration_count=True, tol=1e-5)
print("El sistema lineal fue resuelto en {0} iteraciones".format(it_count))
solution_dirichl_1, solution_neumann_1, solution_dirichl_2 = sol

#Calculo de Energia de Solvatacion

slp_q = bempp.api.operators.potential.laplace.single_layer(neumann_space_1, xq.transpose())
dlp_q = bempp.api.operators.potential.laplace.double_layer(dirichl_space_1, xq.transpose())
phi_q = slp_q*solution_neumann_1 - dlp_q*solution_dirichl_1

Total_energy = 2*np.pi*332.064*np.sum(q*phi_q).real
print("Esolv: {:7.4f} [kcal/mol]".format(Total_energy))
