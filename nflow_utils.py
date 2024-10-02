# -*- coding: utf-8 -*-
"""
utilities for normalizing flows
"""

#%% imports

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.datasets import make_moons


#%% base class for flows

# Base class
class Flow(nn.Module):
    # base class

    def __init__(self):
        super().__init__()
    # end func

    def forward(self, x):
        # Compute f(x) and log |det(jacobian(x))|
        raise NotImplementedError
    # end func

    def inverse(self, y):
        # Compute f^-1(y) and inv log |det(jacobian(y))|
        raise NotImplementedError
    # end func

    def get_inverse(self):
        # Get inverse transformation
        return InverseFlow(self)
    # end func
# end class


class InverseFlow(Flow):
    # Change the forward and inverse transformations

    def __init__(self, base_flow):
        '''
        create the inverse flow from a base flow.

        Input:
            base_flow: flow to reverse
        '''
        super().__init__()
        self.base_flow = base_flow
        if hasattr(base_flow, 'domain'):
            self.codomain = base_flow.domain
        if hasattr(base_flow, 'codomain'):
            self.domain = base_flow.codomain
    # end func

    def forward(self, x):
        '''
        compute the forward transformation given an input x

        Input:
            x: input sample. shape [batch_size, dim]

        Output:
            y: sample after forward tranformation; shape [batch_size, dim]
            log_det_jac: log determinant of the jacobian of the forward tranformation; shape [batch_size]
        '''
        
        y, log_det_jac = self.base_flow.inverse(x)
        return y, log_det_jac
    # end func

    def inverse(self, y):
        '''
        compute the inverse transformation given an input y

        Input:
            y: input sample. shape [batch_size, dim]

        Outut:
            x: sample after inverse tranformation; shape [batch_size, dim]
            inv_log_det_jac: log determinant of the jacobian of the inverse tranformation; shape [batch_size]
        '''
        
        x, inv_log_det_jac = self.base_flow.forward(y)
        return x, inv_log_det_jac
    # end func
# end class


#%% utils to load moons dataset, etc

class CircleGaussiansDataset(Dataset):
    def __init__(self, n_gaussians = 6, n_samples = 100, radius = 3., variance = 0.3, seed = 0):
        '''
        create a 2D dataset with Gaussians on a circle

        Input:
            n_gaussians: number of Gaussians (int)
            n_samples: number of sample per Gaussian (int)
            radius: radius of the circle where the Gaussian means lie (float)
            varaince: varaince of the gaussians (float)
            seed: random seed (int)
        '''
        
        # set params
        self.n_gaussians = n_gaussians
        self.n_samples = n_samples
        self.radius = radius
        self.variance = variance
        # set seed
        np.random.seed(seed)
        
        # make n_gaussians angles, from 0 to 2*pi
        radial_pos = np.linspace(0, np.pi * 2, num = n_gaussians, endpoint = False)
        # get gaussian centers (means) from the angles: x = R*cos(phi), y = R*sin(phi)
        mean_pos = radius * np.column_stack((np.sin(radial_pos), np.cos(radial_pos)))
        
        samples = []
        for ix, mean in enumerate(mean_pos):
            # get n_samples gaussian samples around this center
            # x, y are independent (individual variances, not variance matrix)
            sampled_points = mean[:,None] + (np.random.normal(loc = 0, scale = variance, size = n_samples), np.random.normal(loc = 0, scale = variance, size = n_samples ))
            samples.append(sampled_points)
        # samples is a list of n_gaussian [2, n_samples] arrays
        # permute the collected samples randomly
        p = np.random.permutation(self.n_gaussians * self.n_samples)
        # transpose the samples into (n_gaussians, n_samples, 2) and reshape into (x, 2)
        # finally apply permutation
        self.X = np.transpose(samples, (0, 2, 1)).reshape([-1,2])[p]
    # end func

    def __len__(self):
        return self.n_gaussians * self.n_samples
    # end func

    def __getitem__(self, item):
        # get element with index 'item'
        x = torch.from_numpy(self.X[item]).type(torch.FloatTensor)
        return x
    # end func
# end class
    

class MoonsDataset(Dataset):
    def __init__(self, n_samples = 1200, seed = 0):
        '''
        create a 2D dataset with spirals

        Input:
            n_samples: number of sample per spiral (int)
            seed: random seed (int)
        '''
        self.n_samples = n_samples

        np.random.seed(seed)
        # use the sklearn method to make moons
        self.X, _ = make_moons(n_samples = n_samples, shuffle = True, noise = 0.05, random_state = None)
    # end func

    def __len__(self):
        return self.n_samples
    # end func 
    
    def __getitem__(self, item):
        x = torch.from_numpy(self.X[item]).type(torch.FloatTensor)
        return x
    # end func
# end class


#%% utils for plotting flows

def plot_samples(model, num_samples = 500, mesh_size = 5):
    '''
    plot samples from a normalizing flow model. 
    Colors are selected according to the densities at the samples.

    Input:
        model: normalizing flow model (Flow or StackedFlows)
        num_samples: number of samples to plot (int)
        mesh_size: range for the 2D mesh (float)
    '''
    
    # sample points and their log probs from the flow
    x, log_prob = model.rsample(batch_size = num_samples)
    x = x.cpu().detach().numpy()
    log_prob = log_prob.cpu().detach().numpy()
    prob = np.exp(log_prob)
    
    # make scatter, color coded by prob of that point
    plt.scatter(x[:,0], x[:,1], c = prob)
    plt.xlim(-mesh_size, mesh_size)
    plt.ylim(-mesh_size, mesh_size)
    plt.show()
# end func


def plot_density(model, loader = [], batch_size = 100, mesh_size = 5.0, device = 'cpu'):
    '''
    plot the density of a normalizing flow model. 
    If loader not empty, it plots also its data samples.

    Input:
        model: normalizing flow model: Flow or StackedFlows
        loader: loader containing data to plot: DataLoader
        bacth_size: discretization factor for the mesh: int
        mesh_size: range for the 2D mesh: float
    '''
    
    with torch.no_grad():
        # make a grid of points
        xx, yy = np.meshgrid(np.linspace(- mesh_size, mesh_size, num = batch_size), np.linspace(- mesh_size, mesh_size, num = batch_size))
        coords = np.stack((xx, yy), axis=2)
        # reshape into (x, 2)
        coords_resh = coords.reshape([-1, 2])
        log_prob = np.zeros((batch_size**2))
        
        for i in range(0, batch_size**2, batch_size):
            # get a batch of points from the grid
            data = torch.from_numpy(coords_resh[i:i+batch_size, :]).float().to(device)
            # calculate the log probs of the points
            log_prob[i:i+batch_size] = model.log_prob(data.to(device)).cpu().detach().numpy()
        probs = np.exp(log_prob)
        # make scatter colored according to probs
        plt.scatter(coords_resh[:,0], coords_resh[:,1], c = probs)
        plt.colorbar()
        for X in loader:
            # plot also the samples
            plt.scatter(X[:,0], X[:,1], marker = 'x', c = 'orange', alpha = 0.05)

        plt.show()
    # end with
# end func

