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
#propagate in time:    
for i in range(0,T):
    print(i)
    organ.growth() #Growing one segment
    #########################
    #Configuration of the model for local differental growth vectors:
    omega=0.1                               #Plant Circumnutations' angular frequency
    Vec_Tropism=np.array([1.0,1.0,0.0])     #Direction of constant directional tropism
    n_trop=Vec_Tropism/np.sqrt(Vec_Tropism[0]**2+Vec_Tropism[1]**2+Vec_Tropism[2]**2) #Direction of constant directional tropism
    r_p=np.array([1.0,1.0,1.0])             #Location of attracting point for an attracting point tropism
    kappa_0 =0.1                            #non-zero intrinsic curvature
    ########################
    #Local sensing terms (the shape of the array should be (3,NUM)):
    Delta_constant_tropism = Z*gamma*np.tile(n_trop,(organ.NUM,1)).transpose()   #projection to the N-direction is performed in the update function
    Delta_circumnutations = (R*omega/E) * organ.kappa*organ.D[:,1,:]
    Delta_proprioception = -gamma*organ.kappa*organ.D[:,0,:]
    Delta_intrinsic_curvature = kappa_0*gamma*organ.D[:,0,:]
    r_diff = np.tile(r_p,(organ.NUM,1)).transpose()-organ.r              
    n_diff=np.zeros(np.shape(r_diff))
    for n in range(0,organ.NUM):
        n_diff[:,n]=r_diff[:,n]/np.sqrt(r_diff[0,n]**2+r_diff[1,n]**2+r_diff[2,n]**2)
    Delta_point_tropism = Z*gamma*n_diff #projection to the N-direction is performed in the update function
    #setting the differential growth vector:
    Delta = Delta_constant_tropism + Delta_circumnutations + Delta_proprioception + Delta_intrinsic_curvature
    #########################
    #updating organ:
    #First, we turn the initial coordinate system according to a given vector, 
    #to prevent fast rotations of the base due to misalignement. Should fit the
    #differential growth vector at the base.
    if i==0:
        organ.initial_rotation(Delta[:,0])
    organ.update(Delta,V,E,dt)
###################
#plotting final form:    
plt.close('all')
organ.plot_organ()


