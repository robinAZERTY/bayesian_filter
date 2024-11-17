import numpy as np


"""
predict: 
x = f(x,u)
P = Fx*P*Fx^T + Fu*Cov_u*Fu^T

update:
y = z - h(x)
S = H*P*H^T + R
K = P*H^T*S^-1
x = x + K*y
P = (I - K*H)*P
"""
class Ekf:
    
    def jacobian(self, vector_function, x, y0,  h=1e-4):
        n = len(x)
        m = len(y0)
        J = np.zeros((m,n))
        for i in range(n):
            dx = np.zeros(n)
            dx[i] = h
            y1 = vector_function(x + dx)
            J[:,i] = (y1 - y0).flatten()/h
        return J
    
    
    def __init__(self,x_dim, u_dim = 1):
        self.x_dim = x_dim    
        self.x = np.zeros(x_dim)
        self.P = np.eye(x_dim)*1e4
        self.c = np.array([0])
        self.cov_u = np.zeros((u_dim,u_dim))
        self.Fx = np.eye(x_dim)
        self.Fu = np.zeros((x_dim,u_dim))
        
    def predict(self,f, u):
        prev_x = self.x
        self.x = f(prev_x, u, self.c)
        self.Fx = self.jacobian(lambda x: f(x, u, self.c), prev_x, self.x)
        self.Fu = self.jacobian(lambda uu: f(prev_x, uu, self.c), u, self.x)
        self.P = self.Fx @ self.P @ self.Fx.T + self.Fu @ self.cov_u @ self.Fu.T
        
    def update(self, h, z, R, h_pred = None, S = None):
        h_pred = h(self.x,self.c)
        H = self.jacobian(lambda x: h(x,self.c), self.x, h_pred)
        y = z - h_pred
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.x_dim) - K @ H) @ self.P
        
    def mahalanobis(self, h, z, R, h_pred = None, H = None, S = None):
      
        if h_pred is None:
            h_pred = h(self.x,self.c)
        y = z - h_pred
        
        if H is None:
            H = self.jacobian(h, self.x, h_pred)
        
        if S is None:      
            S = H @ self.P @ H.T + R
            
        return y.T @ np.linalg.inv(S) @ y

#example
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    dt=0.1
    
    def f(X,U,C):
        x = X[0]
        y = X[1]
        theta = X[2]
        v = U[0]
        w = U[1]
        x = x + v*np.cos(theta) * C[0]
        y = y + v*np.sin(theta) * C[0]
        theta = theta + w*C[0]
        return np.array([x,y,theta])
    
    def h(X,C = None):
        return X[:2]
    
    def h2(X,C = None):
        x = X[0]
        y = X[1]
        return np.array([np.sqrt(x**2 + y**2)])
    
    T=20
    
    t = np.arange(0,T,dt)
    
    #generate trajectory from polar coordinates
    r= np.linspace(1,0.6,len(t))**2
    theta = np.linspace(0,4*np.pi,len(t))
    
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    
    direction = np.zeros((2,len(t)))
    direction[0,1:] = (x[1:] - x[:-1])/dt
    direction[1,1:] = (y[1:] - y[:-1])/dt
    direction[0,0] = direction[0,1]
    direction[1,0] = direction[1,1]
    theta = np.arctan2(direction[1,:],direction[0,:])
    
    #compute control inputs
    omega = np.zeros(len(t))
    omega[1:] = (theta[1:] - theta[:-1])/dt
    omega[0] = omega[1]
    velocity = np.sqrt(direction[0,:]**2 + direction[1,:]**2)
    
    X=np.array([x,y,theta])
    
    #simulate noisy measurements
    gpsNoise = 0.05
    z1 = h(X) + np.random.normal(0,gpsNoise,(2,len(t)))
    R1 = gpsNoise**2*np.eye(2)
    
    beaconNoise = 0.05
    z2 = h2(X) + np.random.normal(0,beaconNoise,(1,len(t)))
    R2 = beaconNoise**2*np.eye(1)
    
    #simulate noisy control inputs
    velocityNoise = 0.05
    omegaNoise = 0.1
    U = np.array([velocity+np.random.normal(0,velocityNoise,len(t)),omega+np.random.normal(0,omegaNoise,len(t))])
    Q = np.diag([velocityNoise**2,omegaNoise**2])
    
    
    EkfX = np.zeros((3,len(t)))
    EkfP = np.zeros((3,3,len(t)))

    ekf = Ekf(3)
    ekf.cov_u = Q
    ekf.x = np.array([0,0,theta[0]])
    ekf.c = np.array([dt])

    for i in range(len(t)):
        ekf.predict(f, U[:,i])
        ekf.update(h, z1[:,i], R1)
        ekf.update(h2, z2[:,i], R2)
        EkfX[:,i] = ekf.x
        EkfP[:,:,i] = ekf.P
        
    plt.figure()
    plt.plot(x,y)
    plt.plot(EkfX[0,:],EkfX[1,:])
    plt.scatter(z1[0,:],z1[1,:])
    plt.show()
    

    
    
   