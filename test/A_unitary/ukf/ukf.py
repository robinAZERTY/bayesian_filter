# from uf import UF
import numpy as np
from scipy.linalg import ldl

def cubature_f(f, x_mean, Cov_x, u_mean, Cov_u, c):
    x_dim = len(x_mean)
    u_dim = len(u_mean)
    total_x_dim = x_dim + u_dim
    sqrt__x_dim = np.sqrt(total_x_dim)
    Y = np.zeros((x_dim, 2*total_x_dim)) # transformed sigma points
    Y_mean = np.zeros(x_dim) # transformed mean
    Y_cov = np.zeros((x_dim, x_dim)) # transformed covariance
    Lx, Dx, _ = ldl(Cov_x)
    Lu, Du, _ = ldl(Cov_u)

    LLx = Lx @ np.sqrt(Dx) * sqrt__x_dim
    LLu = Lu @ np.sqrt(Du) * sqrt__x_dim
    for i in range(x_dim):
        x_mean += LLx[:,i]
        Y[:,i] = f(x_mean, u_mean, c)
        x_mean -= 2*LLx[:,i]
        Y[:,i+x_dim] = f(x_mean, u_mean, c)
        x_mean += LLx[:,i]
    for i in range(u_dim):
        u_mean += LLu[:,i]
        Y[:,2*x_dim+i] = f(x_mean, u_mean, c)
        u_mean -= 2*LLu[:,i]
        Y[:,2*x_dim+i+u_dim] = f(x_mean, u_mean, c)
        u_mean += LLu[:,i]
    Y_mean = np.mean(Y, axis=1)
    for i in range(0, Y.shape[1]):
        Y_cov += np.outer(Y[:,i] - Y_mean, Y[:,i] - Y_mean)
    Y_cov /= 2*total_x_dim
    return Y.T, Y_mean, Y_cov

def cubature_h(h, x_mean, Cov_x, c):
    x_dim = len(x_mean)
    z_dim = len(h(x_mean, c))
    total_x_dim = x_dim
    sqrt__x_dim = np.sqrt(total_x_dim)
    Y = np.zeros((z_dim, 2*total_x_dim))
    Y_mean = np.zeros(z_dim)
    Y_cov = np.zeros((z_dim, z_dim))
    Lx, Dx, _ = ldl(Cov_x)
    LLx = Lx @ np.sqrt(Dx) * sqrt__x_dim
    for i in range(x_dim):
        x_mean += LLx[:,i]
        Y[:,i] = h(x_mean, c)
        x_mean -= 2*LLx[:,i]
        Y[:,i+x_dim] = h(x_mean, c)
        x_mean += LLx[:,i]
    Y_mean = np.mean(Y, axis=1)
    for i in range(0, Y.shape[1]):
        Y_cov += np.outer(Y[:,i] - Y_mean, Y[:,i] - Y_mean)
    Y_cov /= 2*total_x_dim
    return Y.T, Y_mean, Y_cov
                 
class Ukf:
    def __init__(self,x_dim, u_dim=1, c_dim=1):
        self.x_dim = x_dim
        self.x = np.zeros(x_dim)
        self.P = np.zeros((x_dim,x_dim))
        self.cov_u = np.zeros((u_dim,u_dim))
        self.c = np.zeros(c_dim)
        self.yx = np.zeros((x_dim, 2*x_dim+1)) # transformed sigma points
        
    def predict(self, f, u):
        self.yx, self.x, self.P = cubature_f(f, self.x, self.P, u, self.cov_u, self.c)
        return self.yx #return the transformed sigma points for visualization
    
    def update(self, h, z, R):
        yy , z_hat, Pzz = cubature_h(h, self.x, self.P, self.c)
        Pz = Pzz + R
        #cross covariance
        Pxz = np.zeros((self.x_dim, z_hat.shape[0]))
        for i in range(2*self.x_dim):
            for j in range(self.x_dim):
                for k in range(z_hat.shape[0]):
                    Pxz[j,k] += (self.yx[i,j] - self.x[j]) * (yy[i,k] - z_hat[k])
            # print((self.yx[:,i]-self.x).reshape(-1,1))
            # print((yy[:,i]-z_hat).reshape(1,-1))
            # print(Pxz)
            # Pxz += (self.yx[:,i]-self.x).reshape(1,-1) * (yy[:,i]-z_hat).reshape(-1,1) 
        Pxz /= 2*self.x_dim * np.sqrt(2)
        K = Pxz @ np.linalg.inv(Pz)
        self.x += K @ (z - z_hat)
        self.P -= Pxz @ K.T
        return yy, z_hat, Pzz, Pxz  #return the transformed sigma points for visualization
        
    def mahalanobis(self, h, z, R):
            yy , z_hat, Pzz = cubature_h(h, self.x, self.P, self.c)
            return (z-z_hat) @ np.linalg.inv(Pzz+R) @ (z-z_hat)
    def __str__(self) -> str:
        return f"UKF: x_dim={self.x_dim}, x={self.x}, P={self.P}, u={self.u}, Q={self.cov_u}"

if __name__ == "__main__": 
    def plot_covariance_ellipse(xEst, PEst, confidence=0.62, N=30):
        #return a list of points to plot the ellipse
        #xEst is the estimated position
        #PEst is the covariance matrix
        xy = xEst[0:2]
        Pxy = PEst[0:2,0:2]
        
        #if one of the diagonal elements of the covariance matrix is infinite, return an empty list
        
        if np.isinf(Pxy).any(axis=None):
            return []
        
    
        #calculate the eigenvalues and eigenvectors of the covariance matrix
        eigVals, eigVecs = np.linalg.eig(Pxy)
        #calculate the angle of the ellipse
        theta = np.arctan2(eigVecs[1,0], eigVecs[0,0])
        #calculate the length of the semi-major and semi-minor axes
        a = np.sqrt(eigVals[0])
        b = np.sqrt(eigVals[1])
        #multiply by the chi-squared value to get the confidence interval
        s= np.sqrt(chi2.ppf(confidence, 2))
        a*=s
        b*=s
        #calculate the points of the ellipse
        t = np.linspace(0, 2*np.pi, N)
        ellipse = np.array([a*np.cos(t) , b*np.sin(t)])
        #rotate the ellipse
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        ellipse = np.dot(R, ellipse)
        #translate the ellipse
        ellipse[0,:] += xy[0]
        ellipse[1,:] += xy[1]
        return ellipse 


    def f(x,u,c=None):
        # return np.array([x[0]**3 + u[0], u[1]+x[1]**2])
        return np.array([x[0]+u[0]*np.cos(u[1]), x[1]+u[0]*np.sin(u[1])])

    def h(x,c=None):
        return np.array([x[0], x[1]])

    X = np.array([0.5, 0.5])
    P = np.array([[0.000001, 0.0], [0.0, 0.000001]])
    U = np.array([1, 0.0])
    Q = np.array([[0.000001, 0.0], [0.0, 0.01]])
    
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    from scipy.stats import chi2
    #monte carlo simulation
    n_samples = 100000
    X_samples = np.random.multivariate_normal(X, P, n_samples)
    U_samples = np.random.multivariate_normal(U, Q, n_samples)
    #compute the covariance of the samples
    fX_samples = f(X_samples.T, U_samples.T).T
    #compute the covariance of the transformed samples
    P_f_samples = np.cov(fX_samples.T)
    #compute the mean of the transformed samples
    fX_mean = np.mean(fX_samples, axis=0)
    
    ukf = Ukf(2, 2)
    ukf.x = X
    ukf.P = P
    ukf.u = U
    ukf.cov_u = Q
    yx = ukf.predict(f, U)
    
    #plot the transformed samples
    plt.scatter(fX_samples[:, 0], fX_samples[:, 1],alpha=1/np.sqrt(n_samples), label='transformed population samples')
    #plot the mean and covariance of the transformed samples
    ellipse = plot_covariance_ellipse(fX_mean, P_f_samples)
    plt.plot(ellipse[0,:], ellipse[1,:], color='red', label='transformed population covariance')
    plt.scatter(fX_mean[0], fX_mean[1], color='red', marker='x', label='transformed population mean')
    ellipse = plot_covariance_ellipse(ukf.x, ukf.P)
    plt.plot(ellipse[0,:], ellipse[1,:], color='green', label='UKF covariance')
    plt.scatter(ukf.x[0], ukf.x[1], color='green', marker='x', label='UKF mean')
    #plot the transformed sigma points (ukf.yx and ukf.yu)
    plt.scatter(yx[:,0], yx[:,1], color='green', marker='o', label='UKF transformed sigma points')
    # plt.scatter(ukf.yu[0,:], ukf.yu[1,:], color='green', marker='o')
    plt.legend()
    plt.show()
    
    # same to test cubature_h
    def h(x,c=None):
        return np.array([(x[0]-0.495)**5, (x[1]-0.495)**3])
    
    R = np.array([[0.000001, 0.0], [0.0, 0.000001]])
    
    hX_samples = h(X_samples.T).T
    P_h_samples = np.cov(hX_samples.T)
    
    hX_mean = np.mean(hX_samples, axis=0)
    
    Y_T, Y_mean, Y_cov = cubature_h(h, X, P, None)
    
    plt.scatter(hX_samples[:, 0], hX_samples[:, 1],alpha=1/np.sqrt(n_samples), label='transformed population samples')
    ellipse = plot_covariance_ellipse(hX_mean, P_h_samples)
    plt.plot(ellipse[0,:], ellipse[1,:], color='red', label='transformed population covariance')
    plt.scatter(hX_mean[0], hX_mean[1], color='red', marker='x', label='transformed population mean')
    ellipse = plot_covariance_ellipse(Y_mean, Y_cov)
    plt.plot(ellipse[0,:], ellipse[1,:], color='green', label='UKF covariance')
    plt.scatter(Y_mean[0], Y_mean[1], color='green', marker='x', label='UKF mean')
    plt.scatter(Y_T[:,0], Y_T[:,1], color='green', marker='o', label='UKF transformed sigma points')
    plt.legend()
    plt.show()