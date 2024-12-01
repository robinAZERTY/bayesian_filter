# from uf import UF
import numpy as np

from scipy.linalg import ldl
class Cubature:
    def __init__(self, f, x_dims, y_dim = None):
        self.x_dims = x_dims
        self.y_dim = y_dim
        self.f = f
        self.total_x_dim = np.sum(x_dims)
        self.sqrt__x_dim = np.sqrt(self.total_x_dim)
        self.Y = np.zeros((self.y_dim, 2*self.total_x_dim)) # transformed sigma points
        self.Y_mean = np.zeros(self.y_dim) # transformed mean
        self.Y_cov = np.zeros((self.y_dim, self.y_dim))
    
    def compute(self, X_mean, P):
        LP, DP, _ = ldl(P[0][0])
        LQ, DQ, _ = ldl(P[1][1])
        LLP = LP @ np.sqrt(DP) * self.sqrt__x_dim
        LLQ = LQ @ np.sqrt(DQ) * self.sqrt__x_dim
        for i in range(self.x_dims[0]):
            X_mean[0] += LLP[:,i]
            self.Y[:,i] = self.f(X_mean)
            X_mean[0] -= 2*LLP[:,i]
            self.Y[:,i+self.x_dims[0]] = self.f(X_mean)
            X_mean[0] += LLP[:,i]
        for i in range(self.x_dims[1]):
            X_mean[1] += LLQ[:,i]
            self.Y[:,2*self.x_dims[0]+i] = self.f(X_mean)
            X_mean[1] -= 2*LLQ[:,i]
            self.Y[:,2*self.x_dims[0]+i+self.x_dims[1]] = self.f(X_mean)
            X_mean[1] += LLQ[:,i]
        
        #compute average
        self.Y_mean = np.mean(self.Y, axis=1)
        # compute covariance
        for i in range(0, self.Y.shape[1]):
            self.Y_cov += np.outer(self.Y[:,i] - self.Y_mean, self.Y[:,i] - self.Y_mean)
        self.Y_cov /= 2*self.total_x_dim
        
        return self.Y.T, self.Y_mean, self.Y_cov
                    
class UKF:
    def __init__(self,f,x_dim, u_dim):
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.x = np.zeros(x_dim)
        self.P = np.zeros((x_dim,x_dim))
        self.cuf = Cubature(lambda x: f(x), [x_dim, u_dim], x_dim)
        self.u = np.zeros(u_dim)
        self.Cov_u = np.zeros((u_dim,u_dim))
        
        
    def predict(self):
        self.yx, self.x, self.P = self.cuf.compute([self.x, self.u], np.array([[self.P, np.zeros((self.x_dim, self.u_dim))], [np.zeros((self.u_dim, self.x_dim)), self.Cov_u]]))
    
    def update(self, h, z, R):
        self.cuh = Cubature(lambda x: h(*x), [self.x_dim], z.shape[0])
        _, z_hat, Pz = self.cuh.compute(np.array([self.x]), np.array([[self.P]]))
        Pz += R
        #cross covariance
        Pxz = np.zeros((self.x_dim, self.x_dim))
        for i in range(2*self.x_dim+1):
            Pxz += self.ufx.sigma_points[:,i] - self.x * self.ufu.sigma_points[:,i] - z_hat     
        K = Pxz @ np.linalg.inv(Pz)
        self.x += K @ (z - z_hat)
        self.P -= K @ Pz @ K.T
        
    def __str__(self) -> str:
        return f"UKF: x_dim={self.x_dim}, x={self.x}, P={self.P}, u={self.u}, Q={self.Cov_u}"

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


    def f(X):
        x = X[0]
        u = X[1]
        # return np.array([x[0]**3 + u[0], u[1]+x[1]**2])
        return np.array([x[0]+u[0]*np.cos(u[1]), x[1]+u[0]*np.sin(u[1])])

    def h(x):
        return np.array([x[0], x[1]])

    X = np.array([0.5, 0.5])
    P = np.array([[0.000001, 0.0], [0.0, 0.000001]])
    U = np.array([1, 0.0])
    Q = np.array([[0.000001, 0.0], [0.0, 0.01]])
    
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    from scipy.stats import chi2
    #monte carlo simulation
    n_samples = 10000
    X_samples = np.random.multivariate_normal(X, P, n_samples)
    U_samples = np.random.multivariate_normal(U, Q, n_samples)
    #compute the covariance of the samples
    fX_samples = f([X_samples.T, U_samples.T]).T
    #compute the covariance of the transformed samples
    P_f_samples = np.cov(fX_samples.T)
    #compute the mean of the transformed samples
    fX_mean = np.mean(fX_samples, axis=0)
    
    ukf = UKF(f, 2, 2)
    ukf.x = X
    ukf.P = P
    ukf.u = U
    ukf.Cov_u = Q
    ukf.predict()
    
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
    plt.scatter(ukf.yx[:,0], ukf.yx[:,1], color='green', marker='o', label='UKF transformed sigma points')
    # plt.scatter(ukf.yu[0,:], ukf.yu[1,:], color='green', marker='o')
    plt.legend()
    plt.show()