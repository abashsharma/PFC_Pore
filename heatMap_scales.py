import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import csv
import math

# Create data
#x = np.random.randn(4096)
#y = np.random.randn(4096)
dim=1024
x=[]
y=[]
for i in range(0,dim):
	x.append(i)
	y.append(i)
z=[]
with open('data.csv') as csvfile:
	plots=csv.reader(csvfile,delimiter=' ')
	
	for row in plots:
		#x.append(int(row[0]))
		#y.append(int(row[1]))
		
		a=float(row[0])
		b=float(row[1])
		c=float(row[2])
		r=(a-511)**2+(b-511)**2
		r=math.sqrt(r)
		if (r<60):
			new_z=-0.195
			z.append(float(new_z))	
		else:
			z.append(c)

densities=np.array(z).reshape(dim,dim)

# Create heatmap
#heatmap, xedges, yedges = np.histogram2d(x, y, bins=(64,64))
#extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

# Plot heatmap
fig=plt.figure()
fig.set_size_inches(4,4)
plt.clf()
plt.pcolormesh(x,y,densities,cmap='jet',vmin=-0.4,vmax=0.1)
plt.axis('off')
plt.show()

fig.savefig('data.png',dpi=600)
