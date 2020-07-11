#Generar archivo config.
import numpy
import sys

Nombre_carpeta_principal = sys.argv[1] #'1PGB'
pqr_file_dato = sys.argv[2] #'1PGB_mut'
prot_file_dato = sys.argv[3] #'mesh/1PGB_mut_p1.5_d02'
tilt_begin = sys.argv[4] #VALORES: 0;38;74;110;146
tilt_end = sys.argv[5] #VALORES: 36;72;108;144;180
H = sys.argv[6] #Valores: 2,3,4,5

name1='_'+str(H)+'_'+str(tilt_begin)+'-'+str(tilt_end)
config_file=Nombre_carpeta_principal+'/'+Nombre_carpeta_principal+'.config'
param_file=Nombre_carpeta_principal+'/'+Nombre_carpeta_principal+'.param'
config_file_moved = config_file[:-7] + name1 + config_file[-7:]
fm = open(config_file_moved, 'w')
conf_file=open(config_file,'r')
for line_full in conf_file:
    line = line_full.split()
    if line[0]=='FILE':
        if line[2]=='dielectric_interface' and line[1]==str(prot_file_dato):
            prot_file = line[1]
            prot_file_moved = prot_file + name1
            new_line = line[0] + '\t' + prot_file_moved + '\t' + line[2] + '\n'
            fm.write(new_line)
        if line[2]=='dielectric_interface' and line[1]!=str(prot_file_dato):
            fm.write(line_full)
        if line[2]=='neumann_surface' or line[2]=='dirichlet_surface':
            surf_file = line[1]
            phi_file = line[3]
            fm.write(line_full)
        
        
    elif line[0]=='FIELD' and line[7]==str(pqr_file_dato)+'.pqr' and int(line[5])>0:
        #pqr_file_aux = line[7]
        pqr_file = pqr_file_dato
        pqr_file_moved = pqr_file + name1 + '.pqr'
        new_line = line[0] + '\t'
        for i in range(1,len(line)):
            if i==7:
                new_line += pqr_file_moved + '\t'
            else:
                new_line += line[i] + '\t'
        fm.write(new_line+'\n')

    else:
        fm.write(line_full)
            
conf_file.close()
fm.close()