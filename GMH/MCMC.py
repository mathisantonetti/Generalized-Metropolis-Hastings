import numpy as np

def Gaussian_distrib(mu, sigma):
    d = mu.shape[0]
    return lambda x:np.exp(-(x-mu)@np.linalg.solve(sigma, x - mu)/2.0)

def banana_distrib(mu, sigma, a, b):
    invbanana = lambda x:np.array([x[0]/a, a*x[1] - a*b*((x[0]/a)**2+a**2)])

    return lambda x:np.exp(-(invbanana(x)-mu)@np.linalg.solve(sigma, invbanana(x) - mu)/2.0)

def banana_sample(a, b, mu, Sigma, N):
    u = np.random.multivariate_normal(mu, Sigma, size=N)
    x = a*u[:, 0]
    y = u[:, 1]/a+b*(u[:, 0]**2+a**2)
    return x, y

def compare_bananas(X, Y, a, b):
    Z1 = np.array([X[:, 0]/a, a*X[:, 1] - a*b*((X[:, 0]/a)**2+a**2)]) # First gaussian
    Z2 = np.array([Y[:, 0]/a, a*Y[:, 1] - a*b*((Y[:, 0]/a)**2+a**2)]) # Second gaussian

    return np.abs(Z1.mean() - Z2.mean()), np.abs(Z1.std() - Z2.std())


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
        return np.prod([np.exp(-(y[i]-self.mu(x))@np.linalg.solve(self.sigma(x), y[i] - self.mu(x))/2.0) for i in range(y.shape[0])])
    
    def sample(self, x, N):
        return np.random.multivariate_normal(self.mu(x), self.sigma(x), size=N)
    

def Generalized_Metropolis_Hastings(distrib, kernel, x0, T, N, M=None):
    if(M is None):
        M = N

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
        for m in range(M):
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
        concat = []
        for i in range(N+1):
            concat.append(np.concatenate((xtilde[1:i], xtilde[i+1:]), axis=0))
        
        A = np.zeros((N+1, N+1))
        for i in range(N+1):
            for j in range(N+1):
                if(j != i):
                    A[i,j] = min(1, distrib(xtilde[i])*kernel(xtilde[i], concat[i])/distrib(xtilde[j])*kernel(xtilde[j], concat[j]))/N
            A[i,i] = 1.0 - np.sum(A[i])

        # Step 3
        for m in range(N):
            I = np.random.choice(N+1, p=A[I])
            x.append(np.copy(xtilde[I]))

    return x