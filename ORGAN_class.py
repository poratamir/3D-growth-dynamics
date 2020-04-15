#%%
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy.linalg import expm
import matplotlib.gridspec as gridspec


class ORGAN:    #contains information about the centerline
    def __init__(self,NUM,NUM_gz,ds,R):      #Starts as a straight organ pointing to the z direction
        self.k1 = 0.0*np.ones((NUM,))        #curvature
        self.k2 = 0.0*np.ones((NUM,))        #curvature
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
        return np.matmul(np.matmul(self.D[:,:,ind],np.array([[0.0,0.0,self.k1[ind]],[0,0,self.k2[ind]],[-self.k1[ind],-self.k2[ind],0]])),self.D[:,:,ind].transpose())
    
    def Update_centerline(self):
        for ind in range(1,self.NUM):       #propgation in arc length
            U=self.Darboux_Matrix(ind-1)      #Darboux skew-symmetric matrix
            self.D[:,:,ind]=np.matmul(expm(self.ds*U),self.D[:,:,ind-1])  #Rotation of the local coordinates using the Rodrigues formula
            self.r[:,ind]=self.r[:,ind-1]+self.ds*self.D[:,2,ind-1]       #centerline update
        self.phi=np.arctan2(self.k2,self.k1)
        self.kappa=np.sqrt(self.k2**2+self.k1**2)
        #normal and bi-normal directions:
        self.N=self.D[:,0,:]*np.cos(self.phi)+self.D[:,1,:]*np.sin(self.phi)
        self.B=self.D[:,0,:]*(-np.sin(self.phi))+self.D[:,1,:]*np.cos(self.phi)
        self.N[:,np.argwhere(self.kappa<1e-14)]=0.0*self.N[:,np.argwhere(self.kappa<1e-10)]
        self.B[:,np.argwhere(self.kappa<1e-14)]=0.0*self.B[:,np.argwhere(self.kappa<1e-10)]

                                        
    def growth(self):   #increasing the length of the organ by one segment
        self.k1=np.append(self.k1,0.0*np.ones((1,)))
        self.k2=np.append(self.k2,0.0*np.ones((1,)))
        self.phi=np.append(self.phi,0.0*np.ones((1,)))
        self.kappa=np.append(self.kappa,0.0*np.ones((1,)))
        self.r=np.append(self.r,np.zeros((3,1)),axis=1)
        self.N=np.append(self.N,np.zeros((3,1)),axis=1)
        self.B=np.append(self.B,np.zeros((3,1)),axis=1)
        self.D=np.append(self.D,np.zeros((3,3,1)),axis=2)
        #initializtion of the new segment:
        U=self.Darboux_Matrix(-2)                                    #Darboux skew-symmetric matrix
        self.D[:,:,-1]=np.matmul(expm(self.ds*U),self.D[:,:,-2])     #Rotation of the local coordinates using the Rodrigues formula
        self.r[:,-1]=self.r[:,-2]+self.ds*self.D[:,2,-2]             #centerline update
        self.NUM+=1                #increase number of segements
        self.k1[-1]=self.k1[-2]    #intializaing curvature
        self.k2[-1]=self.k2[-2]    #intializaing curvature

    
    def update(self,Delta,V,E,ds,dt):
        q=0          #a dummy variable for setting the correct velocity
        for n in range(np.max([self.NUM-self.NUM_gz,0]),self.NUM): #updating the growth zone only
            if n>0: #the base is clamped
                #time step t:
                v=V[q] #velocity of arc-length
                q+=1
                k1_t=(E/self.R)*np.dot(Delta[:,n],self.D[:,0,n])-v*(self.k1[n]-self.k1[n-1])/self.ds  #calculation of curvature's dynamics
                k2_t=(E/self.R)*np.dot(Delta[:,n],self.D[:,1,n])-v*(self.k2[n]-self.k2[n-1])/self.ds  #calculation of curvature's dynamics
                if (np.abs(k1_t)<1e-12): #numerical error threshold
                    k1_t=0.0
                if (np.abs(k2_t)<1e-12): #numerical error threshold
                    k2_t=0.0
                self.k1[n]=self.k1[n]+dt*k1_t
                self.k2[n]=self.k2[n]+dt*k2_t
        #update of the local coordinate systems:           
        self.Update_centerline()


    def plot_organ(self,fig,i,plot_line,plot_point,plot_arrow): #Plotting
        fig.clf()
        gs = gridspec.GridSpec(ncols=5, nrows=2, figure=fig,wspace=0.5, hspace=0.5)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title(r'$\kappa$')
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.set_title(r'$\phi$')
        ax3 = fig.add_subplot(gs[0:, 1:], projection='3d',proj_type = 'ortho')

        s=np.linspace(0.0,self.NUM*self.ds,self.NUM)
        ax1.plot(s,self.kappa,'black',linewidth=2)
        ax2.plot(s,self.phi,'black',linewidth=2)
        ax1.set_xlabel(r'$s$')
        ax2.set_xlabel(r'$s$')
        ax1.set_ylabel(r'$\kappa$')
        ax2.set_ylabel(r'$\phi$')
        ax1.set_frame_on(1)
        ax2.set_frame_on(1)
        
        ax3.clear()
        if plot_line:
            ax3.plot([0,3],[0.3,0.3],[0,3],'-',color="red",alpha=1,linewidth=2)
        if plot_point:
            ax3.plot([1],[-1],[1],'.',color="red",alpha=1.0,markersize=20)
        if plot_arrow:
            ax3.quiver([0],[-self.R*5],[self.R*5],[1.0],[0.0],[0.0],color='black', length=0.5)

        ax3.plot(self.r[0],self.r[1],self.r[2],'black',linewidth=3)
        n1=64.0
        w=range(0,len(self.r[0])-self.NUM_gz)
        for n in range(0,int(n1)):
            L=self.r[:,w]+self.R*np.cos(2*pi*n/n1)*self.D[:,0,w]+self.R*np.sin(2*pi*n/n1)*self.D[:,1,w]
            ax3.plot(L[0],L[1],L[2],'black',alpha=0.1)
        L=self.r[:,w[1:]]+self.R*self.N[:,w[1:]]
        ax3.plot(L[0],L[1],L[2],'blue',alpha=0.75)
        
        w=range(len(self.r[0])-self.NUM_gz,len(self.r[0]))
        apex_plot=np.zeros((3,int(n1+1)))
        for n in range(0,int(n1)):
            L=self.r[:,w]+self.R*np.cos(2*pi*n/n1)*self.D[:,0,w]+self.R*np.sin(2*pi*n/n1)*self.D[:,1,w]
            ax3.plot(L[0],L[1],L[2],'green',alpha=0.1)
            apex_plot[:,n]=[L[0,-1],L[1,-1],L[2,-1]]
        L=self.r[:,w]+self.R*self.N[:,w]
        ax3.plot(L[0],L[1],L[2],'blue',alpha=0.75)
        apex_plot[:,-1]=apex_plot[:,0]
        ax3.plot(apex_plot[0,:],apex_plot[1,:],apex_plot[2,:],'green',alpha=0.5)
        
        jj=-1
        T_dir=self.D[:,2,-1]
        N_dir=self.N[:,-1] 
        B_dir=self.B[:,-1] 
        ax3.quiver(self.r[0,jj],self.r[1,jj],self.r[2,jj],T_dir[0],T_dir[1],T_dir[2],color='red', length=0.5)
        ax3.quiver(self.r[0,jj],self.r[1,jj],self.r[2,jj],N_dir[0],N_dir[1],N_dir[2],color='blue', length=0.5)
        ax3.quiver(self.r[0,jj],self.r[1,jj],self.r[2,jj],B_dir[0],B_dir[1],B_dir[2],color='green', length=0.5)
        l=self.R*5
        ax3.plot([l],[l],[0],alpha=0.0)
        ax3.plot([-l],[-l],[0],alpha=0.0)
        ax3.plot([l],[-l],[0],alpha=0.0)
        ax3.plot([-l],[l],[0],alpha=0.0)

        set_axes_equal(ax3)
        ax3.xaxis.pane.fill = False
        ax3.yaxis.pane.fill = False
        ax3.zaxis.pane.fill = False
        ax3.xaxis.pane.set_edgecolor('gray')
        ax3.yaxis.pane.set_edgecolor('gray')
        ax3.zaxis.pane.set_edgecolor('gray')
        ax3.grid(False)
        ax3.view_init(20, 45-i*0.1)
 


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
    plot_radius = 0.5*max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

