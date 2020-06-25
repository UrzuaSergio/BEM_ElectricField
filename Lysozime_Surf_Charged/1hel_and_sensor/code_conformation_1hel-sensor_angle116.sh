#!/bin/bash

#PBS -q gpuk
#PBS -N 1FZAsurf2d_0_4
#PBS -o 1FZAsurf2d_0_4_out
#PBS -e 1FZAsurf2d_0_4_err
#PBS -l mem=6gb
#PBS -l cput=75:00:00
#PBS -l walltime=80:00:00
#PBS -m bea
#PBS -M sergio.urzua.13@sansano.usm.cl

#use cuda8
#use anaconda2


#export PYTHONPATH=/user/s/surzua/lib/python

cd Lysozime_Surf_Charged/1hel_sensor/
###############################################################################################################################################################
echo "Simulación que contempla las siguientes Orientaciones:"
echo "Tilt begin: 116 - Tilt_end: 116 - Ntilt: 1 -- Rot begin: 0° - Rot end: 360° - Nrot: 36"
echo "...................................................................................."
echo "Proteina-Superficie: Lisozima - Surf Cargada -0.04 C/m^2."
echo "Separación: 0.2 nm"
python generador_config_file_modif.py 1hel_sensor 1hel mesh/1hel_d02 116 116 1

echo "Se ha Creado Config File Auxiliar"
echo "Comenzo Ejecucion de PyGBe para Caso Proteina-Superficie Interactuando"
echo "%%%%%%%% Simulando %%%%%%%%"

python conformation_1hel.py 1hel_sensor 1hel_sensor/1hel 1hel_sensor/mesh/1hel_d04 116 116 1 2 1hel-sensor_116-116-2

echo "Termino Ejecucion de PyGBe para Caso Proteina-Superficie Interactuando"
##############################################################################################################################################################
