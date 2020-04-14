#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
###############################
#Organ class:
import ORGAN_class as OR
###############################
#Varables:
T=300                  #Number of timesteps 
dt=0.1                 #timestep
ds=0.01                #segment length
L0=1.0                 #Total inital length
R=0.1                  #organ's radius
gamma=0.1              #proprioception constant 
lambda0=10*gamma       #tropic sensitivity0                
lambda1=5*gamma        #tropic sensitivity1                
################################
#Initialization:
L_gz=L0                               #apical growth-zone length - equals to inital length 
E=ds/dt/L_gz                          #constant growth rate - one segment each timestep
NUM=int(L0/ds)                        #number of segments
NUM_gz=int(L_gz/ds)                   #number of segments in the growth zone
Vmax=E*NUM_gz*ds                      #apex' velocity
V=np.linspace(0,Vmax-E*ds,NUM)        #Velocity of growth zone
organ=OR.ORGAN(NUM,NUM_gz,ds,R)       #define organ

fig = plt.figure(figsize=(6,4),dpi=200,constrained_layout=False)
plot_arrow=0
plot_line=0
plot_point=0

###################
#propagate in time:    
def growth_timestep(i):
    print(i)
    organ.growth() #Growing one segment
    #########################
    #Setting up differential growth terms
    #(the shape of the arrays should be (3,NUM))
    ########################
    #Proprioception:
    Delta_proprioception = -gamma* organ.kappa*organ.N
    #########################
    #Constant tropism:
    Vec_Tropism=np.array([1.0,0.0,0.0])             #Direction of constant directional tropism
    n_trop=Vec_Tropism/np.sqrt(Vec_Tropism[0]**2+Vec_Tropism[1]**2+Vec_Tropism[2]**2) 
    n_trop=np.tile(n_trop,(organ.NUM,1)).transpose() 
    Delta_constant_tropism = lambda0*n_trop         #projection to the cross-section plane is performed in the update function
    plot_arrow=1
    #########################
    #Point tropism:
    r_p=np.array([1.0,-1.0,1.0])                    #Location of an attracting point for point tropism
    r_diff = np.tile(r_p,(organ.NUM,1)).transpose()-organ.r              
    n_diff=np.zeros(np.shape(r_diff))
    for n in range(0,organ.NUM):
        n_diff[:,n]=r_diff[:,n]/np.sqrt(r_diff[0,n]**2+r_diff[1,n]**2+r_diff[2,n]**2)
    Delta_point_tropism = lambda0*n_diff           #projection to the cross-section plane is performed in the update function
    plot_point=1
    #########################
    #Twining:
    r_rod=np.array([0.0,0.3,0.0])                   #Location of an attracting line for twining
    r_rod=np.tile(r_rod,(organ.NUM,1)).transpose()
    n_rod=np.array([1.0,0.0,1.0])/np.sqrt(2)        #Direction of attracting line for twining
    n_rod=np.tile(n_rod,(organ.NUM,1)).transpose()
    n_trop2=n_rod                                   #Direction of constant directional tropism in twining
    rho=organ.r-r_rod
    rho=rho-n_rod*(rho[0,:]*n_rod[0,:]+rho[1,:]*n_rod[1,:]+rho[2,:]*n_rod[2,:])
    rho=np.divide(rho,np.sqrt(rho[0,:]**2+rho[1,:]**2+rho[2,:]**2))
    Delta_twining = lambda0*n_trop2 -lambda1*rho
    plot_line=1
    #########################
    #Circumnutation:
    omega=0.2                                       #Plant Circumnutations' angular frequency
    Delta_circumnutations = lambda1*(organ.D[:,0,:]*np.cos(i*omega)+organ.D[:,1,:]*np.sin(i*omega))
    ########################
    ########################
    #setting the differential growth vector (choose scenario):
    Delta = Delta_constant_tropism + Delta_proprioception
    Delta = Delta_point_tropism    + Delta_proprioception
    Delta = Delta_twining          + Delta_proprioception
    Delta = Delta_circumnutations  + Delta_proprioception
    Delta = Delta_constant_tropism + Delta_circumnutations + Delta_proprioception
    #########################
    #updating organ:
    organ.update(Delta,V,E,ds,dt)
    organ.plot_organ(fig,-1*i,plot_line,plot_point,plot_arrow)
    return []
###################
#saving animation output: 
NAME="NAME"
NAME_tot="C:\\Users\\amirp\\Documents\\Python Scripts\\3D dynamics\\new_anim_april\\" + NAME
def init():
    return []
anim = animation.FuncAnimation(fig, growth_timestep, init_func=init,frames=T, interval=30, blit=True)
anim.save(NAME_tot+'.mp4', fps=20, extra_args=['-vcodec', 'libx264'])
plt.close('all')
#%%saving final organ:
import pickle
f = open(NAME_tot+'.pckl', 'wb')
pickle.dump(organ, f)
f.close()
