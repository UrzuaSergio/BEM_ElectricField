import numpy
import sys
import os
import glob

#ACA SE DEBE COLOCAR EL NOMBRE DE LA CARPETA QUE CONTIENE TODOS LOS ARCHIVOS, 
#LOS NOMBRES DE LOS ARCHIVOS ORIGINALES DE EL ARCHIVO.PQR Y LOS ARCHIVOS.MESH


Nombre_carpeta_principal = sys.argv[1] 
pqr_file = sys.argv[2] 
prot_file = sys.argv[3] 
tilt_begin = sys.argv[4] 
tilt_end = sys.argv[5] 
tilt_N = sys.argv[6] 
H = sys.argv[7] #Valores: 2,3,4,5
output_file = sys.argv[8] 


name1='_'+str(H)+'_'+str(tilt_begin)+'-'+str(tilt_end)
param_file=Nombre_carpeta_principal+'/'+Nombre_carpeta_principal+'.param'
config_file=Nombre_carpeta_principal+'/'+Nombre_carpeta_principal+'.config'
config_file_moved = config_file[:-7] + name1 + config_file[-7:]

##########################################################################################################
##########################################################################################################

file_config_datos=open(config_file_moved,'r')
for line in file_config_datos:
	line1=line.split()
	
	
	if line1[0]=='FILE':
		if line1[2]== 'dielectric_interface':
			surf_file='NA'
			phi_file='NA'
		if line1[2]== 'neumann_surface':
			surf_file=Nombre_carpeta_principal+'/'+line1[1]
			phi_file= line1[3]
       	        
file_config_datos.close()

#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################

#Funcion para leer archivo_output

def scanOutput(filename):
    
    flag = 0 
    files = []
    file1=open(filename,'r')
    for line in file1:
        line = line.split()
        if len(line)>0:
            if line[0]=='Converged':
                iterations = int(line[2])
            if line[0]=='Total' and line[1]=='elements':
                N = int(line[-1])
            if line[0]=='Totals:':
                flag = 1 
            if line[0]=='E_solv' and flag==1:
                Esolv = float(line[2])
            if line[0]=='E_surf' and flag==1:
                Esurf = float(line[2])
            if line[0]=='E_coul' and flag==1:
                Ecoul = float(line[2])
            if line[0]=='Time' and flag==1:
                Time = float(line[2])
            if line[0]=='Reading':
               files.append(line[-1])  

    return N, iterations, Esolv, Esurf, Ecoul, Time, files

#########################################################################################################

N = []
iterations = []
Esolv = []
Esurf = []
Ecoul = []
Time = []


#########################################################################################################
#Generacion de Angulos (Orientacion)

til_min = float(tilt_begin)
til_max = float(tilt_end)
til_N = int(tilt_N)  

rot_min = 0. #valor es 0
rot_max = 360. # valor es 360 ;Non-inclusive end point
rot_N = 1  #valor es 36 por ejemplo

til_angles_aux = numpy.linspace(til_min, til_max, num=til_N)  # Tilt angles (inclusive end point)
rot_angles_aux = numpy.linspace(rot_min, rot_max, num=rot_N, endpoint=False)  # Rotation angles


til_angles = []
rot_angles = []
for i in range(len(til_angles_aux)):
    if abs(til_angles_aux[i])<1e-10 or abs(til_angles_aux[i]-180)<1e-10:
        til_angles.append(til_angles_aux[i])
        rot_angles.append(rot_min)
    else:
        for j in range(len(rot_angles_aux)):
            til_angles.append(til_angles_aux[i])
            rot_angles.append(rot_angles_aux[j])


###################################################################################################################################################################
###################################################################################################################################################################
for i in range(len(til_angles)):
    #mover proteina en distintos estados y generar archivos pqr y mesh. se guardan en carpeta PRINCIPAL Y EN mesh/ ----> ESTA BIEN
    name='_'+str(H)+'_'+str(tilt_begin)+'-'+str(tilt_end)
    
    cmd_move = 'python move_protein_1hel.py ' + prot_file + ' ' + pqr_file + ' ' + str(int(rot_angles[i])) + ' ' + str(int(til_angles[i])) +' '+ str(H) + ' ' + name
    os.system(cmd_move)
    
    #ejecutar pygbe
    cmd_run = ' /home/sergiourzua/bin/pygbe '+ Nombre_carpeta_principal +' -p' + param_file + ' -c' + config_file_moved +' > output_aux_' + output_file + name
    os.system(cmd_run)

    N_run, iterations_run, Esolv_run, Esurf_run, Ecoul_run, Time_run, files = scanOutput('output_aux_' + output_file + name)

    print('rot_angle: ',rot_angles[i],'til_angle: ',til_angles[i])
   

    N.append(N_run)
    iterations.append(iterations_run)
    Esolv.append(Esolv_run)
    Esurf.append(Esurf_run)
    Ecoul.append(Ecoul_run)
    Time.append(Time_run)

   

Etotal = numpy.array(Esolv) + numpy.array(Esurf) + numpy.array(Ecoul)
EsurfEsolv = numpy.array(Esolv) + numpy.array(Esurf)

fout = open(output_file, 'w')

fout.write('\nParameter file:\n')
f = open(param_file, 'r')
fout.write(f.read())
fout.write('\n')
f.close()

fout.write('Protein file: ' + prot_file + '\n')
fout.write('Sensor  file: ' + surf_file + '\n')
fout.write('Phi     file: ' + phi_file + '\n\n')

fout.write('\nNumber of elements  : %i \n'%(N[0]))
fout.write('Number of iterations: max: %i, min: %i, avg: %i \n'%(max(iterations), min(iterations), int(numpy.average(iterations))))
fout.write('Coulombic energy    : %f kcal/mol \n'%(Ecoul[0]))
fout.write('Total time          : max: %fs, min: %fs, avg: %fs \n' %(max(Time), min(Time), numpy.average(Time)))

fout.write('\n                    ||              kcal/mol\n')
fout.write('  h     | Tilt   |  Rotat  ||    Esolv      |    Esurf     |      Esurf+Esolv \n')
fout.write('------------------------------------------------------------------------------ \n')
for i in range(len(til_angles)):
    fout.write('  %3.2f  |  %3.2f  |  %3.2f || %s  | %s    | %s \n'%(float(H),til_angles[i], rot_angles[i], Esolv[i], Esurf[i], EsurfEsolv[i]))
fout.close()
###################################################################################################################################################################


os.system('rm -r output_aux_' + output_file + name)
os.system('rm -r ' + config_file_moved)
os.system('rm -r ' + pqr_file + name + '.pqr')
os.system('rm -r ' + prot_file + name + '.vert')
os.system('rm -r ' + prot_file + name + '.face')





















