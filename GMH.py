import numpy as np

def Gaussian_distrib(mu, sigma):
    d = mu.shape[0]
    return lambda x:np.exp(-(x-mu)@np.linalg.solve(sigma, x - mu))/np.sqrt(2*np.pi)**(d/2)

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
        xtilde[1:, :] = kernel.sample(xtilde[I], N)
        
        # Step 2
        pIcond = np.ones(d)

        piK_I = distrib(xtilde[I])*kernel(xtilde[I], np.concatenate(xtilde[:I], xtilde[I+1:], axis=0))
        for j in range(d):
            if(j != I):
                pIcond[j] = min(1, distrib(xtilde[j])*kernel(xtilde[j], np.concatenate(xtilde[:j], xtilde[j+1:], axis=0))/piK_I)/N
                pIcond[I] += pIcond[j]
        pIcond[I] = 1 - pIcond[I]

        # Step 3
        for m in range(N):
            I = np.random.choice(0, N+1, p=pIcond)
            x.append(xtilde[I])

    return x