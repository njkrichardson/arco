'''
Created on 14 May 2013
@author: James McInerney

Edited on 13 June 2020 
@author: njkr 
'''
import numpy as np 
import numpy.random as npr
import matplotlib.pyplot as plt
from scipy.special import digamma
from tqdm import tqdm 
from viz import create_cov_ellipse

import sys 
import os 
import shutil

# ENV 
RESULTS_DIR = '/Users/nickrichardson/Desktop/personal/projects/picard/tests/comparators/gmm_animations'
npr.seed(2305)

def simulate(n_components : int = 30, n_obs : int = 200, obs_dim : int = 2) -> np.ndarray:
    """
    Generate samples from a mixture of Gaussians. 

    Parameters
    ----------- 
    n_components : int 
        number of component distributions (default 30)
    n_obs : int 
        number of obserations to sample from the model (default 200)
    obs_dim : int 
        observation dimension (default 2)
    """ 
    mu = np.array([npr.multivariate_normal(np.zeros(obs_dim), 10 * np.eye(obs_dim)) for _ in range(n_components)])
    cov = [0.1 * np.eye(obs_dim) for _ in range(n_components)]
    pi = npr.dirichlet(np.ones(n_components))              # mixture distribution -- pi ~ Dir(1/k)
    inputs = np.zeros((n_obs, obs_dim))                    # observation model -- x_i | z_i ~ Normal(mu_i, cov_i)
    z = np.zeros((n_obs, n_components))                    # latent assignments -- z_i | pi ~ Categorical(pi)
    for i in range(n_obs):
        z[i] = npr.multinomial(1, pi)
        k = np.argmax(z[i])
        inputs[i] =  npr.multivariate_normal(mu[k], cov[k])
    return inputs, np.argmax(z, axis=1)

def run(inputs : np.ndarray, n_components : int, hyperpriors : dict, verbose : bool = True, save_every : int = 5 , max_iterations : int = 200):
    """
    Experiment wrapper. 

    Parameters
    ----------
    inputs : np.ndarray
        observations
    n_components : int 
        number of mixture components 
    verbose : bool
        verbosity (default True)
    """
    # observations 
    (n_obs, obs_dim) = inputs.shape 
    
    # initialize variational parameters
    q_z = np.array([npr.dirichlet(np.ones(n_components)) for _ in range(n_obs)])

    # initialize plot (if relevant)
    if verbose is True: 
        plt.ion()    
        fig = plt.figure(figsize=(10, 10))
        ax_spatial = fig.add_subplot(1, 1, 1) #http://stackoverflow.com/questions/3584805/in-matplotlib-what-does-111-means-in-fig-add-subplot111
        circs = []
                
    for i in tqdm(range(max_iterations), disable=(not verbose)):
        # M step: update the "global" parameters 
        N_k = q_z.sum(axis=0)
        beta_k = hyperpriors['beta_0'] + N_k
        inv_wishart_dof = hyperpriors['inv_wishart_dof'] + N_k + 1.
        weighted_means = get_weighted_means(q_z, inputs, N_k)
        covs = update_covs(q_z, inputs, weighted_means, N_k)
        means = update_means(n_components, obs_dim, hyperpriors['beta_0'], hyperpriors['prior_mean'], N_k, weighted_means, beta_k)
        precisions = update_precisions(n_components, hyperpriors['prior_precision'], weighted_means, N_k, hyperpriors['prior_mean'], obs_dim, hyperpriors['beta_0'], covs)

        # E step: update the "local" variational distribution parameters
        energy = get_energy(inputs, obs_dim, N_k, beta_k, means, precisions, inv_wishart_dof, n_obs, n_components) #eqn 10.64 Bishop
        log_det_precision = get_log_det_precisions(precisions, inv_wishart_dof, obs_dim, n_components) #eqn 10.65 Bishop
        expected_log_pi = get_expected_log_pi(hyperpriors['mixture_concentration'], N_k) #eqn 10.66 Bishop
        q_z = update_variational_params(obs_dim, expected_log_pi, log_det_precision, energy, n_obs, n_components) #eqn 10.46 Bishop (really 10.46 taken through 10.49)
        
        if verbose is True: 
            if i == 0:
                plt.scatter(inputs[:, 0], inputs[:, 1])
                sctZ = plt.scatter(means[:, 0], means[:, 1], color='r')
            elif (i % save_every == 0):
                #ellipses to show covariance of components
                for circ in circs: circ.remove()
                circs = []
                for k in range(n_components):
                    circ = create_cov_ellipse(covs[k], means[k], color='r', alpha=0.3) #calculate params of ellipses (adapted from http://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals)
                    circs.append(circ)
                    #add to axes:
                    ax_spatial.add_artist(circ)
                    #make sure components with N_k=0 are not visible:
                    if N_k[k]<= hyperpriors['mixture_concentration']: means[k] = means[N_k.argmax(),:] #put over point that obviously does have assignments
                sctZ.set_offsets(means)
                plt.draw()
                plt.savefig(os.path.join(RESULTS_DIR, f"{i}.png"))
    
    if verbose is True:
        pass
    
    return means, log_det_precision, expected_log_pi, q_z
    
    
def get_weighted_means(q_z : np.ndarray, inputs : np.ndarray, N_k : np.ndarray) -> np.ndarray:
    (n_obs, obs_dim) = inputs.shape 
    (_, n_components) = q_z.shape
    weighted_means = np.zeros((n_components, obs_dim))
    
    for i in range(n_obs):
        for k in range(n_components):
            weighted_means[k] += q_z[i][k] * inputs[i]
    
    for k in range(n_components):
        if N_k[k] > 0: 
            weighted_means[k] /= N_k[k]

    return weighted_means

def update_covs(q_z : np.ndarray, inputs : np.ndarray, weighted_means : np.ndarray, N_k : np.ndarray) -> list:
    (n_obs, n_components) = q_z.shape
    (_, obs_dim) = inputs.shape
    
    covs = [np.zeros((obs_dim, obs_dim)) for _ in range(n_components)]
    for i in range(n_obs):
        for k in range(n_components):
            B0 = (inputs[i] - weighted_means[k]).reshape(obs_dim, 1)
            L = np.dot(B0, B0.T)
            covs[k] += q_z[i, k] * L

    for k in range(n_components):
        if N_k[k] > 0: 
            covs[k] /= N_k[k]
    return covs

def update_precisions(n_components : int, prior_precision : float, weighted_means : np.ndarray, N_k : np.ndarray, prior_mean : np.ndarray, obs_dim : int, beta_0 : float, covs : list) -> list:
    covs_new = [None for _ in range(n_components)]
    for k in range(n_components): 
        covs_new[k]  = np.linalg.inv(prior_precision) + N_k[k] * covs[k]
        mean_dist = (weighted_means[k] - prior_mean).reshape(obs_dim, 1)
        sample_cov = np.dot(mean_dist, mean_dist.T)
        covs_new[k] += (beta_0 * N_k[k] / (beta_0 + N_k[k]) ) * sample_cov

    precisions = []
    for k in range(n_components):
        try:
            precisions.append(np.linalg.inv(covs_new[k]))
        except linalg.linalg.LinAlgError as linalg_error:
            print('Invalid update to precision matrix (uninvertible covariance)')
            raise linalg_error
    return precisions

def update_means(n_components : int, obs_dim : int, beta_0 : float, prior_mean : np.ndarray, N_k : np.ndarray, weighted_means : np.ndarray, beta_k : float) -> np.ndarray:
    means = np.zeros((n_components, obs_dim))
    for k in range(n_components): 
        means[k] = (beta_0 * prior_mean + N_k[k] * weighted_means[k]) / beta_k[k]
    return means 

def get_energy(inputs : np.ndarray, obs_dim : int, N_k : np.ndarray, beta_k : float, means : np.ndarray, precisions : list, inv_wishart_dof : int, n_obs : int, n_components : int) -> np.ndarray:
    energy = np.zeros((n_obs, n_components))
    for i in range(n_obs):
        for k in range(n_components):
            A = obs_dim / beta_k[k] #shape: (k,)
            B0 = (inputs[i] - means[k]).reshape(obs_dim, 1)
            B1 = np.dot(precisions[k], B0)
            l = np.dot(B0.T, B1)
            energy[i][k] = A + inv_wishart_dof[k] * l  #shape: (n,k)
    
    return energy

def get_expected_log_pi(mixture_concentration : float, N_k : np.ndarray) -> np.ndarray:
    mixture_concentration += N_k
    expected_log_pi = digamma(mixture_concentration) - digamma(mixture_concentration.sum())
    return expected_log_pi

def get_log_det_precisions(precisions : list, inv_wishart_dof : int, obs_dim : int, n_components : int) -> list:
    log_det_precisions = [None for _ in range(n_components)]
    for k in range(n_components):
        det_precision = np.linalg.det(precisions[k])
        if det_precision > 1e-30: 
            log_det = np.log(det_precision)
        else: 
            log_det = 0.0
        log_det_precisions[k] = np.sum([digamma((inv_wishart_dof[k] + 1 - i) / 2.) for i in range(obs_dim)]) + obs_dim * np.log(2) + log_det
    return log_det_precisions
        
def update_variational_params(obs_dim : int, expected_log_pi : np.ndarray, log_det_precisions : list, energy : np.ndarray, n_obs : int, n_components : int) -> np.ndarray:
    ln_q_z = np.zeros((n_obs, n_components))  # ln q_z 
    for k in range(n_components):
        ln_q_z[:, k] = expected_log_pi[k] + 0.5 * log_det_precisions[k] - 0.5 * obs_dim * np.log(2 * np.pi) - 0.5 * energy[:, k]

    #normalise ln Z:
    ln_q_z -= ln_q_z.max(axis=1).reshape(n_obs, 1)
    q_z = np.exp(ln_q_z) / np.exp(ln_q_z).sum(axis=1).reshape(n_obs, 1)
    return q_z
    
if __name__ == "__main__":
    # setup results dirs
    shutil.rmtree(RESULTS_DIR, ignore_errors=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    #generate synthetic data
    inputs, z = simulate() 

    # modelling hyperpriors 
    n_components = 20 
    obs_dim = 2 
    hyperpriors = {
        'mixture_concentration': 0.1,                     
        'beta_0': (1e-20),  
        'inv_wishart_dof': obs_dim + 1., 
        'prior_mean': np.zeros(obs_dim), 
        'prior_precision': np.eye(obs_dim)
        }

    # run experiment 
    means, log_det_precisions, expected_log_pi, q_z = run(inputs, n_components, hyperpriors)
    