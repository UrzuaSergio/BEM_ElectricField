"""
Creates a sphere of radius r, centered at x0, y0, z0

Parameters (command line arguments)
----------
rec : (int) number of recursions for unit sphere
r   : (float) radius
x0  : (float) x center of sphere
y0  : (float) y center of sphere
z0  : (float) z center of sphere
name: (str) output file name
 
Output
-----
File with vertices ".vert"
File with triangle indices ".face"
"""
import sys
import numpy
from triangulation import create_unit_sphere

#rec = 4 #512 
#rec = 6 #int(sys.argv[1])#8192
#rec = 5 #2048
rec = 6
#rec = 7 #32768
r = 4.00000000 #float(sys.argv[2])
x0 = 0.00000000 #float(sys.argv[3])
y0 = 0.00000000 #float(sys.argv[4])
z0 = 0.00000000 #float(sys.argv[5])
filename = "sphere_R4" #sys.argv[6]

xc = numpy.array([x0,y0,z0])
vertex, index, center = create_unit_sphere(rec)
vertex *= r
vertex += xc

index += 1 # Agrees with msms format
index_format = numpy.zeros_like(index)
index_format[:,0] = index[:,0]
index_format[:,1] = index[:,2]
index_format[:,2] = index[:,1]

# Check
x_test = numpy.average(vertex[:,0])
y_test = numpy.average(vertex[:,1])
z_test = numpy.average(vertex[:,2])
if abs(x_test-x0)>1e-12 or abs(y_test-y0)>1e-12 or abs(z_test-z0)>1e-12:
    print('Center is not right!')

numpy.savetxt(filename+'.vert', vertex, fmt='%.8f')
numpy.savetxt(filename+'.face', index_format, fmt='%i')

print('Sphere with %i faces, radius %f and centered at %f,%f,%f was saved to the file '%(len(index), r, x0, y0, z0)+filename)
