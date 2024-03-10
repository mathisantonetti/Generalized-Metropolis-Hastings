import numpy as np

def Gaussian_distrib(mu, sigma):
    d = mu.shape[0]
    return lambda x:np.exp(-(x-mu)@np.linalg.solve(sigma, x - mu))/np.sqrt(2*np.pi)**(d/2)

def ESS(data):
    n, m = data.shape

    variogram = lambda t: ((data[t:,:] - data[:(n - t),:])**2).sum() / (m*(n - t))
    # Computing within-chain variance with Numpy
    data_mean = data.mean(axis=1, keepdims=True)
    W = ((data - data_mean)**2).sum()/m
    B = np.sum((data_mean - data_mean.mean(keepdims=True))**2)/(m-1)
    s2 = B + (W/n)

    negative_autocorr = False
    t = 1
    rho = np.ones(n)

    # And another loop!
    while not negative_autocorr and (t < n):
        rho[t] = 1 - variogram(t) / (2 * s2)
        if not t % 2:
            negative_autocorr = sum(rho[t-1:t+1]) < 0
        t += 1

    return n/(1+2*np.sum(rho[1:t]))



class GaussianKernel():
    def __init__(self, mu, sigma):
        self.sigma = sigma
        self.mu = mu

    def __call__(self, x, y):
        return np.prod([np.exp(-(y[i]-self.mu(x))@np.linalg.solve(self.sigma(x), y[i] - self.mu(x)))/np.sqrt(2*np.pi)**(x.shape[0]/2) for i in range(y.shape[0])])
    
    def sample(self, x, N):
        return np.random.multivariate_normal(self.mu(x), self.sigma(x), size=N)
    

def Generalized_Metropolis_Hastings(distrib, kernel, x0, T, N):
    d = x0.shape[0]
    xtilde = np.zeros((N+1, d))
    x = []
    
    # initialization
    xtilde[0, :] = x0
    I = 0

    # MCMC loop
    for t in range(T):
        # Step 1
        new_xtilde = kernel.sample(xtilde[I], N)
        xtilde[:I, :] = new_xtilde[:I, :]
        xtilde[I+1:, :] = new_xtilde[I:, :]
        
        # Step 2
        pIcond = np.ones(N+1)
        for j in range(N+1):
            pIcond[j] = distrib(xtilde[j])*kernel(xtilde[j], np.concatenate((xtilde[:j], xtilde[j+1:]), axis=0))

        pIcond = pIcond/np.sum(pIcond)

        # Step 3
        for m in range(N):
            I = np.random.choice(N+1, p=pIcond)
            x.append(np.copy(xtilde[I]))

    return x

def Truly_Generalized_Metropolis_Hastings(distrib, kernel, x0, T, N):
    d = x0.shape[0]
    xtilde = np.zeros((N+1, d))
    x = []
    
    # initialization
    xtilde[0, :] = x0
    I = 0

    # MCMC loop
    for t in range(T):
        # Step 1
        new_xtilde = kernel.sample(xtilde[I], N)
        xtilde[:I, :] = new_xtilde[:I, :]
        xtilde[I+1:, :] = new_xtilde[I:, :]
        
        # Step 2
        pIcond = np.zeros(N+1)
        piK_I = distrib(xtilde[I])*kernel(xtilde[I], np.concatenate((xtilde[:I], xtilde[I+1:]), axis=0))
        for j in range(N+1):
            if(j != I):
                pIcond[j] = min(1, distrib(xtilde[j])*kernel(xtilde[j], np.concatenate((xtilde[:j], xtilde[j+1:]), axis=0))/piK_I)/N
                pIcond[I] += pIcond[j]

        pIcond[I] = 1 - pIcond[I]

        # Step 3
        for m in range(N):
            I = np.random.choice(N+1, p=pIcond)
            x.append(np.copy(xtilde[I]))

    return x