import numpy as np
import matplotlib.pyplot as pyplot
import torch
from sklearn.cluster import KMeans

class ParetoPP(object):
  def __init__(self, X, num_dims=2, initial_dims=50, perplexity=30.0):
    self.X = X # Input data, N x D array
    self.num_dims = num_dims # Number of dimensions after reduction, scalar
    self.initial_dims = initial_dims # Number of dimensions before reduction, scalar
    self.perplexity = perplexity # Perplexity for tSNE parameter search, scalar

  def Hbeta_torch(self, D, beta=1.0):
    P = torch.exp(-D.clone() * beta)

    sumP = torch.sum(P)

    H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
    P = P / sumP

    return H, P

  def x2p_torch(self, tol=1e-5):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """
    X = self.X # Input data, N x D array
    perplexity = self.perplexity # Perplexity for tSNE parameter search, scalar

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape

    sum_X = torch.sum(X*X, 1)
    D = torch.add(torch.add(-2 * torch.mm(X, X.t()), sum_X).t(), sum_X)

    P = torch.zeros(n, n)
    beta = torch.ones(n, 1)
    logU = torch.log(torch.tensor([perplexity]))
    n_list = [i for i in range(n)]

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        # there may be something wrong with this setting None
        betamin = None
        betamax = None
        Di = D[i, n_list[0:i]+n_list[i+1:n]]

        (H, thisP) = self.Hbeta_torch(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while torch.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].clone()
                if betamax is None:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].clone()
                if betamin is None:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = self.Hbeta_torch(Di, beta[i])

            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, n_list[0:i]+n_list[i+1:n]] = thisP

    # Return final P-matrix
    return P

  def pca_torch(self):

    X = self.X
    num_dims = self.initial_dims

    print("Preprocessing the data using PCA...")
    
    (n, d) = X.shape
    X = X - torch.mean(X, 0)

    (l, M) = torch.eig(torch.mm(X.t(), X), True)
    # split M real
    for i in range(d):
        if l[i, 1] != 0:
            M[:, i+1] = M[:, i]
            i += 1

    Y = torch.mm(X, M[:, 0:num_dims])
    return Y

  def tsne(self):
    """
        Runs t-SNE on the dataset in the N x D array X to reduce its
        dimensionality to num_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, num_dims, perplexity), where X is an NxD NumPy array.
    """
    X = self.X # Input data, N x D array
    num_dims = self.num_dims # Number of dimensions after reduction, scalar

    # Check inputs
    if isinstance(num_dims, float):
        print("Error: array X should not have type float.")
        return -1
    if round(num_dims) != num_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = self.pca_torch()
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = torch.randn(n, num_dims)
    dY = torch.zeros(n, num_dims)
    iY = torch.zeros(n, num_dims)
    gains = torch.ones(n, num_dims)

    # Compute P-values
    P = self.x2p_torch(tol=1e-5)
    P = P + P.t()
    P = P / torch.sum(P)
    P = P * 4.    # early exaggeration
    print("get P shape", P.shape)
    P = torch.max(P, torch.tensor([1e-21]))

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = torch.sum(Y*Y, 1)
        num = -2. * torch.mm(Y, Y.t())
        num = 1. / (1. + torch.add(torch.add(num, sum_Y).t(), sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / torch.sum(num)
        Q = torch.max(Q, torch.tensor([1e-12]))

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = torch.sum((PQ[:, i] * num[:, i]).repeat(num_dims, 1).t() * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum

        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)).double() + (gains * 0.8) * ((dY > 0.) == (iY > 0.)).double()
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - torch.mean(Y, 0)

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = torch.sum(P * torch.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.
        
        self.Y = Y # Data after dimension reduction
    # Return solution
    return Y

  def kmeans(self, n_clusters=2):
    km_results = KMeans(n_clusters=2, random_state=0).fit(self.Y)
    km_labels = km_results.labels_
    return km_labels
