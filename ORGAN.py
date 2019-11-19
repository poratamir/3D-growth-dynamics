#%%
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy.linalg import expm
import matplotlib.gridspec as gridspec


class ORGAN:    #contains information about the centerline
    def __init__(self,NUM,NUM_gz,ds,R):      #Starts as a straight organ pointing to the z diretion
        self.kappa = 0.0*np.ones((NUM,))     #curvature
        self.tau = 0.0*np.ones((NUM,))       #torsion
        self.phi = 0.0*np.ones((NUM,))       #twist angle
        self.r=np.zeros((3,NUM))             #centerline
        self.D=np.zeros((3,3,NUM))           #rotation matrices to centerline's local coordiates
        self.NUM=NUM                         #Number of segments
        self.NUM_gz=NUM_gz                   #Number of segments in the growth-zone
        self.ds=ds                           #segment length
        self.R=R                             #organ's radius
        #centerline initializtion:
        self.D[:,:,0]=np.array([[1,0,0],[0,1,0],[0,0,1]])
        self.Update_centerline()

   
    def Darboux_Matrix(self,ind):   #calculation of the Darboux skew-symmetric matrix in the local coordinate system
        return np.matmul(np.matmul(self.D[:,:,ind],np.array([[0,-self.tau[ind],self.kappa[ind]],[self.tau[ind],0,0],[-self.kappa[ind],0,0]])),self.D[:,:,ind].transpose())
    
    def Update_centerline(self):
        for ind in range(1,self.NUM):       #propgation in arc length
            U=self.Darboux_Matrix(ind-1)      #Darboux skew-symmetric matrix
            self.D[:,:,ind]=np.matmul(expm(self.ds*U),self.D[:,:,ind-1])  #Rotation of the local coordinates using the Rodrigues formula
            self.r[:,ind]=self.r[:,ind-1]+self.ds*self.D[:,2,ind-1]       #centerline update

        
    def initial_rotation(self,vec):   #turning the initial coordinate system according to a given vector
        VEC=vec
        VEC[2]=0                                             #projection to the (N,B) plane
        VEC=VEC/np.sqrt(VEC[0]**2+VEC[1]**2)                 #normalization
        self.D[:,:,0]=np.array([[1,0,0],[0,1,0],[0,0,1]])    #inital base's unit vectors
        self.D[:,:,0]=np.matmul(expm(np.arccos(VEC[0])*np.array([[0,-1,0],[1,0,0],[0,0,0]])),self.D[:,:,0]) #Rotation of the base using the Rodrigues formula
        self.Update_centerline()
                                
    def growth(self):   #increasing the length by one vertebra
        self.kappa=np.append(self.kappa,0.0*np.ones((1,)))
        self.phi=np.append(self.phi,0.0*np.ones((1,)))
        self.tau=np.append(self.tau,0.0*np.ones((1,)))
        self.r=np.append(self.r,np.zeros((3,1)),axis=1)
        self.D=np.append(self.D,np.zeros((3,3,1)),axis=2)
        #initializtion of the new segment, assuming properties from the convection term alone (no differential growth):
        U=self.Darboux_Matrix(-2)                                    #Darboux skew-symmetric matrix
        self.D[:,:,-1]=np.matmul(expm(self.ds*U),self.D[:,:,-2])     #Rotation of the local coordinates using the Rodrigues formula
        self.r[:,-1]=self.r[:,-2]+self.ds*self.D[:,2,-2]             #centerline update
        self.NUM+=1                      #increase number of segements
        self.kappa[-1]=self.kappa[-2]    #intializaing curvature
        self.phi[-1]=self.phi[-2]        #intializaing twist angle
        
    
    def update(self,Delta,V,E,dt):
        q=0          #a dummy variable for setting the correct velocity
        for n in range(np.max([self.NUM-self.NUM_gz,0]),self.NUM): #updating the growth zone only
            if n>0: #the base is clamped
                #time step t:
                v=V[q] #velocity of arc-length
                q+=1
                kappa_t=(E/self.R)*np.dot(Delta[:,n],self.D[:,0,n])-v*(self.kappa[n]-self.kappa[n-1])/self.ds  #calculation of curvature's dynamics
                if np.abs(self.kappa[n])>1e-8:  #calculation of the twist angle's dynamics
                    phi_t=(E/self.R/self.kappa[n])*np.dot(Delta[:,n],self.D[:,1,n])-v*(self.phi[n]-self.phi[n-1])/self.ds
                else:
                    phi_t=0.0-v*(self.phi[n]-self.phi[n-1])/self.ds
                #time step t+dt:
                self.kappa[n]=self.kappa[n]+dt*kappa_t
                self.phi[n]=self.phi[n]+phi_t*dt
                if self.phi[n]<1e-12: #avoiding numerical errors:
                    self.phi[n]=0.0
                self.tau[n]=(self.phi[n]-self.phi[n-1])/self.ds #calculation of torsion
        #update of the local coordinate systems:           
        self.Update_centerline()


    def plot_organ(self): #Plotting
        fig = plt.figure(figsize=(10,6),dpi=200)
        gs = gridspec.GridSpec(ncols=5, nrows=2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title(r'$\kappa$')
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.set_title(r'$\phi$')
        ax3 = fig.add_subplot(gs[0:, 1:], projection='3d')
        plt.tight_layout()
        
        s=np.linspace(0.0,self.NUM*self.ds,self.NUM)
        ax1.plot(s,self.kappa,'r',linewidth=2)
        ax2.plot(s,self.phi,'b',linewidth=2)
        ax1.set_xlabel(r'$s$')
        ax2.set_xlabel(r'$s$')
        ax1.set_ylabel(r'$\kappa$')
        ax2.set_ylabel(r'$\phi$')
        ax1.set_frame_on(1)
        ax2.set_frame_on(1)
        plt.tight_layout()
        
        ax3.clear()
        ax3.plot(self.r[0],self.r[1],self.r[2],'black',linewidth=3)
        n1=16.0
        w=range(0,len(self.r[0])-self.NUM_gz)
        for n in range(0,int(n1)):
            L=self.r[:,w]+self.R*np.cos(2*pi*n/n1)*self.D[:,0,w]+self.R*np.sin(2*pi*n/n1)*self.D[:,1,w]
            if n==0:
                ax3.plot(L[0],L[1],L[2],'black',alpha=0.5)
            else:
                ax3.plot(L[0],L[1],L[2],'black',alpha=0.3)
        w=range(len(self.r[0])-self.NUM_gz,len(self.r[0]))
        apex_plot=np.zeros((3,int(n1+1)))
        for n in range(0,int(n1)):
            L=self.r[:,w]+self.R*np.cos(2*pi*n/n1)*self.D[:,0,w]+self.R*np.sin(2*pi*n/n1)*self.D[:,1,w]
            if n==0:
                ax3.plot(L[0],L[1],L[2],'green',alpha=0.5)
                apex_plot[:,n]=[L[0,-1],L[1,-1],L[2,-1]]
            else:
                ax3.plot(L[0],L[1],L[2],'green',alpha=0.3)
                apex_plot[:,n]=[L[0,-1],L[1,-1],L[2,-1]]
        apex_plot[:,-1]=apex_plot[:,0]
        ax3.plot(apex_plot[0,:],apex_plot[1,:],apex_plot[2,:],'green',alpha=0.5)
        
        lw=2
        l1=0.5
        jj=-1
        T_dir=self.D[:,2,-1]
        N_dir=self.D[:,0,-1]
        B_dir=self.D[:,1,-1]
        ax3.plot([self.r[0,jj],self.r[0,jj]+l1*B_dir[0]],[self.r[1,jj],self.r[1,jj]+l1*B_dir[1]],[self.r[2,jj],self.r[2,jj]+l1*B_dir[2]],'green',alpha=1,linewidth=lw)
        ax3.plot([self.r[0,jj],self.r[0,jj]+l1*N_dir[0]],[self.r[1,jj],self.r[1,jj]+l1*N_dir[1]],[self.r[2,jj],self.r[2,jj]+l1*N_dir[2]],'blue',alpha=1,linewidth=lw)
        ax3.plot([self.r[0,jj],self.r[0,jj]+l1*T_dir[0]],[self.r[1,jj],self.r[1,jj]+l1*T_dir[1]],[self.r[2,jj],self.r[2,jj]+l1*T_dir[2]],'red',alpha=1,linewidth=lw)
        
        set_axes_equal(ax3)
        ax3.xaxis.pane.fill = False
        ax3.yaxis.pane.fill = False
        ax3.zaxis.pane.fill = False
        ax3.xaxis.pane.set_edgecolor('gray')
        ax3.yaxis.pane.set_edgecolor('gray')
        ax3.zaxis.pane.set_edgecolor('gray')
        ax3.grid(False)
        ax3.view_init(30, -45)
        plt.tight_layout()
 


def set_axes_equal(ax):

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

