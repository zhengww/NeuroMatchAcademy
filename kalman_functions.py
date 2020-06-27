# Install PyKalman
# pip install pykalman
import numpy as np
import matplotlib.pyplot as plt
#from pykalman import KalmanFilter
from scipy.stats import multivariate_normal


# Data Visualiztion
def plot_kalman(x, y, nx, ny, kx=None, ky=None, color="r-", label=None, title='LDS', ax=None):
    """
    Plot the trajectory
    """
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(15, 4))
        plotdata=True
    else:
        plotdata=False
    if kx is not None and ky is not None:
        if plotdata:
            ax[0].plot(x, y, 'g-', label='true latent')
            ax[0].plot(nx, ny, 'k.', label='data')
        ax[0].plot(kx, ky, color, label=label)
        #ax[0].plot(kx[0], ky[0], 'or')
        #ax[0].plot(kx[-1], ky[-1], 'xr')

        ax[1].plot(x, kx, '.', color=color, label='latent dim 1')
        ax[1].plot(y, ky, 'x', color=color, label='latent dim 2')
        ax[1].set_xlabel('real latent')
        ax[1].set_ylabel('estimated latent')
        ax[1].legend()
    else:
        ax[0].plot(x, y, 'g-', label='true latent')
        ax[0].plot(nx, ny, 'k.', label='data')
        ax[1].plot(x, nx, '.k', label='dim 1')
        ax[1].plot(y, ny, '.', color='grey', label='dim 2')
        ax[1].set_xlabel('latent')
        ax[1].set_ylabel('observed')
        ax[1].legend()

    ax[0].set_xlabel('X position')
    ax[0].set_ylabel('Y position')
    ax[0].set_title(title)
    ax[0].set_aspect(1)
    ax[1].set_aspect(1)
    ax[1].set_title('correlation')

    ax[0].legend()
    return  ax


def visualize_line_plot(data, xlabel, ylabel, title):
    """
    Function that visualizes a line plot
    """
    plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def print_parameters(kf_model, need_params=None, evals=False):
    """
    Function that prints out the parameters for a Kalman Filter
    @param - kf_model : the model object
    @param - need_params : a list of string
    """
    if evals:
        if need_params is None:
            need_params1 = ['transition_matrices', 'transition_covariance', 'observation_covariance',
                            'initial_state_covariance']
            need_params2 = ['observation_matrices', 'initial_state_mean']
        for param in need_params1:
            tmp = np.linalg.eig(getattr(kf_model, param))[0]
            print("{0} (shape = {2})\n   {1}\n".format(param, tmp, tmp.shape))
        for param in need_params2:
            print("{0} (shape = {2})\n   {1}\n".format(param, getattr(kf_model, param), getattr(kf_model, param).shape))
    else:
        if need_params is None:
            need_params = ['transition_matrices', 'observation_matrices', 'transition_covariance',
                           'observation_covariance',
                           'initial_state_mean', 'initial_state_covariance']
        for param in need_params:
            print("{0} (shape = {2})\n   {1}\n".format(param, getattr(kf_model, param), getattr(kf_model, param).shape))


class MyKalmanFilter:
    """
    Class that implements the Kalman Filter
    """

    def __init__(self, n_dim_state=2, n_dim_obs=2):
        """
        @param n_dim_state: dimension of the laten variables
        @param n_dim_obs: dimension of the observed variables
        """
        self.n_dim_state = n_dim_state
        self.n_dim_obs = n_dim_obs
        self.transition_matrices = np.eye(n_dim_state)
        self.transition_covariance = np.eye(n_dim_state)
        self.observation_matrices = np.eye(n_dim_obs, n_dim_state)
        self.observation_covariance = np.eye(n_dim_obs)
        self.initial_state_mean = np.zeros(n_dim_state)
        self.initial_state_covariance = np.eye(n_dim_state)

    def sample(self, n_timesteps, initial_state=None, random_seed=None):
        """
        Method that gives samples
        @param initial_state: numpy array whose length == self.n_dim_state
        @param random_seed: an integer, for test purpose
        @output state: a 2d numpy array with dimension [n_timesteps, self.n_dim_state]
        @output observation: a 2d numpy array with dimension [n_timesteps, self.n_dim_obs]
        """
        # set initial states and seed
        if initial_state is None:
            initial_state = self.initial_state_mean
        if random_seed is not None:
            np.random.seed(random_seed)

        ################
        ##### TODO #####
        ################
        # produce samples
        latent_state = []
        observed_state = []
        current_latent_state = initial_state
        for t in range(n_timesteps):
            # for the first latent state is set to the initial state
            if t == 0:
                latent_state.append(current_latent_state)
            # otherwise use transition_matrices and transition_covariance to calculate next latent state
            else:
                latent_state.append(np.dot(self.transition_matrices, current_latent_state) +
                                    np.random.multivariate_normal(np.zeros(self.n_dim_state),
                                                                  self.transition_covariance))
                current_latent_state = latent_state[-1]
            # use observation_matrices and observation_covariance to calculate next observed state
            observed_state.append(np.dot(self.observation_matrices, current_latent_state) +
                                  np.random.multivariate_normal(np.zeros(self.n_dim_obs), self.observation_covariance))

        #latent_state = np.zeros([n_timesteps, self.n_dim_state])
        #observed_state = np.zeros([n_timesteps, self.n_dim_obs])
        return np.array(latent_state), np.array(observed_state)

    def filter(self, X, use_myfilter=False):
        """

        Parameters
        ----------
        X :
        use_myfilter :

        Returns
        -------

        """
        """
        Method that performs Kalman filtering
        @param X: a numpy 2D array whose dimension is [n_example, self.n_dim_obs]
        @output: filtered_state_means: a numpy 2D array whose dimension is [n_example, self.n_dim_state]
        @output: filtered_state_covariances: a numpy 3D array whose dimension is [n_example, self.n_dim_state, self.n_dim_state]
        """

        # validate inputs
        n_example, observed_dim = X.shape
        assert observed_dim == self.n_dim_obs

        # create holders for outputs
        filtered_state_means = np.zeros([n_example, self.n_dim_state])
        filtered_state_covariances = np.zeros([n_example, self.n_dim_state, self.n_dim_state])

        #############################
        # TODO: implement filtering #
        #############################
        if use_myfilter:
            # the first state mean and state covar is the initial expectation
            filtered_state_means[0] = self.initial_state_mean
            filtered_state_covariances[0] = self.initial_state_covariance

            # initialize internal variables
            current_state_mean = self.initial_state_mean.copy()
            current_state_covar = self.initial_state_covariance.copy()
            self.p_n_list = np.zeros((n_example, self.n_dim_obs, self.n_dim_obs))
            for i in range(1, n_example):
                current_observed_data = X[i, :]
                # run a single step forward filter
                # prediction step
                predicted_state_mean = np.dot(self.transition_matrices, current_state_mean)
                predicted_state_cov = np.matmul(np.matmul(self.transition_matrices, current_state_covar),
                                                np.transpose(self.transition_matrices)) + self.transition_covariance
                # observation step
                innovation = current_observed_data - np.dot(self.observation_matrices, predicted_state_mean)
                innovation_covariance = np.matmul(np.matmul(self.observation_matrices, predicted_state_cov),
                                                  np.transpose(self.observation_matrices)) + self.observation_covariance
                # update step
                kalman_gain = np.matmul(np.matmul(predicted_state_cov, np.transpose(self.observation_matrices)),
                                        np.linalg.inv(innovation_covariance))
                current_state_mean = predicted_state_mean + np.dot(kalman_gain, innovation)
                current_state_covar = np.matmul((np.eye(current_state_covar.shape[0]) -
                                                 np.matmul(kalman_gain, self.observation_matrices)),
                                                predicted_state_cov)
                # populate holders
                filtered_state_means[i, :] = current_state_mean
                filtered_state_covariances[i, :, :] = current_state_covar
                self.p_n_list[i, :, :] = predicted_state_cov
                # self.p_n_list[i-1, :, :] = predicted_state_cov
            # new
            # self.p_n_list[-1, :, :] = np.matmul(np.matmul(self.transition_matrices, filtered_state_covariances[-1,:,:]),
            #                                    np.linalg.inv(self.transition_matrices)) + self.transition_covariance

        else:
            #################################################################################
            # below: this is an alternative if you do not have an implementation of filtering
            kf = KalmanFilter(n_dim_state=self.n_dim_state, n_dim_obs=self.n_dim_obs)
            need_params = ['transition_matrices', 'observation_matrices', 'transition_covariance',
                           'observation_covariance', 'initial_state_mean', 'initial_state_covariance']
            for param in need_params:
                setattr(kf, param, getattr(self, param))
            filtered_state_means, filtered_state_covariances = kf.filter(X)
        #################################################################################

        return filtered_state_means, filtered_state_covariances

    def smooth(self, X, use_mysmoother=False, use_myfilter=False):
        """
        Method that performs the Kalman Smoothing
        @param X: a numpy 2D array whose dimension is [n_example, self.n_dim_obs]
        @output: smoothed_state_means: a numpy 2D array whose dimension is [n_example, self.n_dim_state]
        @output: smoothed_state_covariances: a numpy 3D array whose dimension is [n_example, self.n_dim_state, self.n_dim_state]
        """
        # TODO: implement smoothing

        # validate inputs
        n_example, observed_dim = X.shape
        assert observed_dim == self.n_dim_obs

        # run the forward path
        mu_list, v_list = self.filter(X, use_myfilter=use_myfilter)

        # create holders for outputs
        smoothed_state_means = np.zeros((n_example, self.n_dim_state))
        smoothed_state_covariances = np.zeros((n_example, self.n_dim_state, self.n_dim_state))

        #############################
        # TODO: implement smoothing #
        #############################
        if use_mysmoother:
            # init for EM
            self.j_n = []

            # last time step doesn't need to be updated
            smoothed_state_means[-1, :] = mu_list[-1, :]
            smoothed_state_covariances[-1, :, :] = v_list[-1, :, :]

            # run the backward path
            # it's zero-indexed and we don't need to update the last elements
            for i in range(n_example - 2, -1, -1):
                # used to store intermediate results
                p_i = np.copy(self.p_n_list[i + 1, :, :])
                # ALTERNATIVELY compute new:
                # p_i = np.matmul(np.matmul(self.transition_matrices, v_list[i,:,:]), self.transition_matrices.T) + self.transition_covariance

                j_i = np.matmul(np.matmul(v_list[i, :, :], self.transition_matrices.T), np.linalg.inv(p_i))

                # calculate mu_bar and v_bar
                current_smoothed_mean = mu_list[i, :] + np.matmul(j_i, (
                        smoothed_state_means[i + 1, :] - np.matmul(self.transition_matrices, mu_list[i, :])))
                current_smoothed_covar = v_list[i, :] + np.matmul(
                    np.matmul(j_i, (smoothed_state_covariances[i + 1, :, :] - p_i)), j_i.T)
                # propagate the holders
                smoothed_state_means[i, :] = current_smoothed_mean
                smoothed_state_covariances[i, :, :] = current_smoothed_covar
                # note that j_n is REVERSELY propagated from N-2 to 0 (zero-indexed)
                self.j_n.append(j_i)
            # add the last j_n
            p_N = np.matmul(np.matmul(self.transition_matrices, v_list[-1, :, :]),
                            np.linalg.inv(self.transition_matrices)) + self.transition_covariance
            j_N = np.matmul(np.matmul(v_list[-1, :, :], self.transition_matrices.T), np.linalg.inv(p_N))
            self.j_n = list(reversed(self.j_n))
            self.j_n.append(j_N)
        else:
            #################################################################################
            # below: this is an alternative if you do not have an implementation of smoothing
            kf = KalmanFilter(n_dim_state=self.n_dim_state, n_dim_obs=self.n_dim_obs)
            need_params = ['transition_matrices', 'observation_matrices', 'transition_covariance',
                           'observation_covariance', 'initial_state_mean', 'initial_state_covariance']
            for param in need_params:
                setattr(kf, param, getattr(self, param))
            _, _ = kf.filter(X)
            smoothed_state_means, smoothed_state_covariances = kf.smooth(X)
            #################################################################################

        return smoothed_state_means, smoothed_state_covariances

    def em(self, X, max_iter=10, use_myfilter=False, use_mysmooth=False):
        """
        This part is OPTIONAL
        Method that perform the EM algorithm to update the model parameters
        Note that in this exercise we ignore offsets
        @param X: a numpy 2D array whose dimension is [n_example, self.n_dim_obs]
        @param max_iter: an integer indicating how many iterations to run
        """
        # validate inputs have right dimensions
        n_example, observed_dim = X.shape
        assert observed_dim == self.n_dim_obs

        # keep track of log posterior (use function calculate_posterior below)
        self.avg_em_log_posterior = np.zeros(max_iter) * np.nan

        #############################
        #### TODO: EM iterations ####
        #############################

        for iter_num in range(max_iter):
            ### Expectation Step ###
            # run the forward and backward path
            if use_mysmooth==False: _, v_list = self.filter(X)
            smoothed_state_means, smoothed_state_covariances = self.smooth(X, use_mysmoother=use_mysmooth,
                                                                           use_myfilter=use_myfilter)

            if use_mysmooth==False:
                ###### if pykalman implementation was used, j_n needs to be computed separatedly #####
                self.j_n = []
                for i in range(n_example - 2, -1, -1):
                    p_i = np.matmul(np.matmul(self.transition_matrices, v_list[i, :, :]),
                                    self.transition_matrices.T) + self.transition_covariance
                    j_i = np.matmul(np.matmul(v_list[i, :, :], self.transition_matrices.T), np.linalg.inv(p_i))
                    self.j_n.append(j_i)
                self.j_n = list(reversed(self.j_n))
                p_N = np.matmul(np.matmul(self.transition_matrices, v_list[-1, :, :]),
                                np.linalg.inv(self.transition_matrices)) + self.transition_covariance
                j_N = np.matmul(np.matmul(v_list[-1, :, :], self.transition_matrices.T), np.linalg.inv(p_N))
                self.j_n.append(j_N)

            self.avg_em_log_posterior[iter_num] = np.nanmean(self.calculate_posterior(X, smoothed_state_means))

            # propagate E[z_n], E[z_n z_{n-1}^T], E[z_n z_n^T]
            self.e_zn = []
            self.e_zn_znminus = []
            self.e_zn_zn = []
            for i in range(n_example):
                self.e_zn.append(smoothed_state_means[i])
                self.e_zn_zn.append(smoothed_state_covariances[i] +
                                    np.outer(smoothed_state_means[i].T, smoothed_state_means[i]))
                # E[z_n z_{n-1}^T] only has n-1 elements
                if i != 0:
                    self.e_zn_znminus.append(np.matmul(self.j_n[i - 1], smoothed_state_covariances[i]) +
                                             np.outer(smoothed_state_means[i], smoothed_state_means[i - 1].T))

            ### Maximization Step ###
            # update initial states and initial covariance
            self.initial_state_mean = smoothed_state_means[0, :]
            self.initial_state_covariance = smoothed_state_covariances[0, :]

            # update transition matrix and transition covariance
            ezy = np.sum(np.array(self.e_zn_znminus), 0)
            ezz = np.sum(np.array(self.e_zn_zn), 0)
            ezz_minus_n = ezz - self.e_zn_zn[-1]
            ezz_minus_1 = ezz - self.e_zn_zn[0]
            self.transition_matrices = np.matmul(ezy, np.linalg.inv(ezz_minus_n))
            ezy_a = np.matmul(ezy, self.transition_matrices.T)
            self.transition_covariance = (ezz_minus_1 - ezy_a - ezy_a.T +
                                          np.matmul(np.matmul(self.transition_matrices, ezz_minus_n),
                                                    self.transition_matrices.T)) / (n_example - 1)

            # update emission matrix and emission covariance
            # x_zn = np.matmul(X.T, np.array(self.e_zn))
            x_zn = np.dot(X.T, np.array(self.e_zn))
            self.observation_matrices = np.matmul(x_zn, np.linalg.inv(ezz.T))
            self.observation_covariance = np.zeros((self.n_dim_obs, self.n_dim_obs))
            for t in range(n_example):
                err = (X[t] - np.dot(self.observation_matrices, smoothed_state_means[t]))
                self.observation_covariance += (np.outer(err, err) + np.dot(self.observation_matrices,
                                                                            np.dot(smoothed_state_covariances[t],
                                                                                   self.observation_matrices.T)))
            self.observation_covariance /= n_example

    def import_param(self, kf_model):
        """
        Method that copies parameters from a trained Kalman Model
        @param kf_model: a Pykalman object
        """
        need_params = ['transition_matrices', 'observation_matrices', 'transition_covariance',
                       'observation_covariance', 'initial_state_mean', 'initial_state_covariance']
        for param in need_params:
            setattr(self, param, getattr(kf_model, param))

    def calculate_posterior(self, X, state_mean, v_n=None):
        """
        Method that calculates the log posterior
        @param X: a numpy 2D array whose dimension is [n_example, self.n_dim_obs]
        @param state_mean: a numpy 2D array whose dimension is [n_example, self.n_dim_state]
        @output: a numpy 1D array whose dimension is [n_example]
        """
        if v_n is None:
            _, v_n = self.filter(X)
        llh = []
        for i in range(1, len(state_mean)):
            normal_mean = np.dot(self.observation_matrices, np.dot(self.transition_matrices, state_mean[i - 1]))
            p_n = self.transition_matrices.dot(v_n[i].dot(self.transition_matrices)) + self.transition_covariance
            # normal_cov = np.matmul(self.observation_matrices, np.matmul(self.p_n_list[i], self.observation_matrices.T)) + self.observation_covariance
            normal_cov = np.matmul(self.observation_matrices,
                                   np.matmul(p_n, self.observation_matrices.T)) + self.observation_covariance
            pdf_val = multivariate_normal.pdf(X[i], normal_mean, normal_cov)
            # replace 0 to prevent numerical underflow
            if pdf_val < 1e-10:
                pdf_val = 1e-10
            llh.append(np.log(pdf_val))
        return np.array(llh)
