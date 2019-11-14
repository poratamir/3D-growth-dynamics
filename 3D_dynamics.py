#%%
import numpy as np
import ORGAN as OR
import matplotlib.pyplot as plt

###############################
#Varables:
T=100               #Number of timesteps 
dt=0.1              #timestep
ds=0.01             #segment length
L0=1.0              #Total inital length
R=0.1               #organ's radius
gamma=0.01          #proprioception constant 
Z=5                 #ratio between tropic sensitivity and proprioception                 
################################
#Initialization:
L_gz=L0                               #apical growth-zone length - equals to inital length 
E=ds/dt/L_gz                          #constant growth rate - one segment each timestep
NUM=int(L0/ds)                        #number of segments
NUM_gz=int(L_gz/ds)                   #number of segments in the growth zone
Vmax=E*NUM_gz*ds                      #apex' velocity
V=np.linspace(0,Vmax-E*ds,NUM)        #Velocity of growth zone
organ=OR.ORGAN(NUM,NUM_gz,ds,R)       #define organ

###################
#turning the initial coordinate system according to a given vector, 
#to prevent fast rotations of the base due to misalignement. Should fit the
#differential growth vector at the base.
organ.initial_rotation(np.array([1.0,0.0,0.0])) 
###################
#propagate in time:    
for i in range(0,T):
    print(i)
    organ.growth() #Growing one segment
    #model for local differental growth vectors: the size of the array should be (3,NUM):
    omega=0.1
    Delta = Z*gamma*(np.tile(np.array([1.0,0.0,0.0]),(organ.NUM,1)).transpose())-gamma*organ.kappa*organ.D[:,0,:]+(R*omega/E) * organ.kappa*organ.D[:,1,:]
    #updating organ:
    organ.update(Delta,V,E,dt)
###################
#plotting final form:    
plt.close('all')
organ.plot_organ()


