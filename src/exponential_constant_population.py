import pymc as pm
import numpy as np
import pytensor.tensor as pt
import arviz as az


# -- The model --
# From 0 to t0
def S_ut_piece1(alpha, Ne1, t, u):
    t = pt.as_tensor_variable(t)[:, None]  # Shape (n_quad, 1)
    u = pt.as_tensor_variable(u)[None, :]  # Shape (1, n_points)
    # If alpha is not close to zero
    inner1 = (1 - pt.exp(alpha * t)) / (2 * Ne1 * alpha)
    exponent1 = alpha * t - 2 * t * u + inner1
    res1 = pt.exp(exponent1) / (2 * Ne1)  # Shape (n_quad, n_points)
    # If alpha is close to zero we use Taylor series
    numerator = 4 * Ne1 + alpha * t * (4 * Ne1 - t)
    exponent2 = -t * (4 * Ne1 * u + 1) / (2 * Ne1)
    res2 = numerator * pt.exp(exponent2) / (8 * Ne1**2)
    epsilon = 1e-5
    return pt.switch(pt.abs(alpha) < epsilon, res2, res1)

# From t0 to infinity
def S_ut_piece2(alpha, Ne1, Ne2, t0, t, u):
    t = pt.as_tensor_variable(t)[:, None]  # Shape (n_quad, 1)
    u = pt.as_tensor_variable(u)[None, :]  # Shape (1, n_points)
    # If alpha is not close to zero
    inner1 = (Ne1 * alpha * (t0 - t) + Ne2 * (1 - pt.exp(alpha * t0))) / (
        2 * Ne1 * Ne2 * alpha
    )
    exponent1 = -2 * t * u + inner1
    res1 = pt.exp(exponent1) / (2 * Ne2)  # Shape (n_quad, n_points)
    # If alpha is close to zero we use Taylor series
    inner2 = 4 * Ne1 - alpha * t0**2
    exponent2 = (-4 * Ne1 * Ne2 * t * u + Ne1 * (t0 - t) - Ne2 * t0) / (2 * Ne1 * Ne2)
    res2 = inner2 * pt.exp(exponent2) / (8 * Ne1 * Ne2)
    epsilon = 1e-5
    return pt.switch(pt.abs(alpha) < epsilon, res2, res1)

def expected_S(u_col, legendre_x, legendre_w, Ne1, Ne2, alpha, t0):
    u_col = pt.as_tensor_variable(u_col)
    # First integral: [0, t0]
    times1 = (t0 - 0) / 2 * legendre_x + (t0 + 0) / 2
    f_t_piece1 = S_ut_piece1(alpha, Ne1, times1, u_col)  # (n_quad, n_points)
    integral_piece1 = pt.sum(
        f_t_piece1 * legendre_w[:, None] * (t0 - 0) / 2, axis=0
    )  # (n_points,)

    # Second integral: [t0, ∞)
    trans_legendre_x = 0.5 * legendre_x + 0.5
    trans_legendre_w = 0.5 * legendre_w
    times2 = t0 + trans_legendre_x / (1 - trans_legendre_x)
    f_t_piece2 = S_ut_piece2(alpha, Ne1, Ne2, t0, times2, u_col)
    integral_piece2 = pt.sum(
        f_t_piece2 * (trans_legendre_w[:, None] / (1 - trans_legendre_x)[:, None] ** 2),
        axis=0,
    )  # (n_points,)
    return integral_piece1 + integral_piece2  # shape (n_points,)

def gauss(a, b, n=10):
    x, w = np.polynomial.legendre.leggauss(n)
    w = (b - a) / 2 * w
    x = (b - a) / 2 * x + (a + b) / 2
    return x, w

def correct_ld(mu, sample_size):
    S = sample_size * 2
    beta = 1 / (S - 1) ** 2
    alpha = (S**2 - S + 2) ** 2 / (S**2 - 3 * S + 2) ** 2
    return (alpha - beta) * mu + 4 * beta

def predict_ld_builder(u_i, u_j, granularity_time = 100, granularity_bins = 10):
    """
    This function returns another function that is compatible with PyMC and
    that will solve the LD values using Gaussian-Legendre rules efficiently.

    Arguments:
    TODO
    """
    legendre_x, legendre_w = np.polynomial.legendre.leggauss(granularity_time)
    u_points = np.array([gauss(a, b, granularity_bins)[0] for (a, b) in zip(u_i, u_j)])
    u_weights = np.array([gauss(a, b, granularity_bins)[1] / (b - a) for (a, b) in zip(u_i, u_j)])
    def expected_ld(u_i, u_j, Ne1, Ne2, alpha, t0):
        u_i = pt.as_tensor_variable(u_i)
        u_j = pt.as_tensor_variable(u_j)
        # I'm not sure this is allowed or causes a big overhead ...
        r2_flat = expected_S(
            u_points.flatten(),
            legendre_x, legendre_w,
            Ne1, Ne2, alpha, t0
        )
        r2_matrix = r2_flat.reshape(u_points.shape)
        return  pt.sum(r2_matrix * u_weights, axis=1)
    def predict_ld(Ne1, Ne2, alpha, t0, sample_size):
        return correct_ld(expected_ld(u_i, u_j, Ne1, Ne2, alpha, t0), sample_size)
    return predict_ld

# -- Arviz wrapper for LOO --
class ConstantPyMCModelWrapper(az.PyMCSamplingWrapper):
    def sample(self, modified_observed_data):
        with self.model:
            # if the model had coords the dim needs to be updated before
            # modifying the data in the model with set_data
            # otherwise, we don't need to overwrite the sample method
            n__i = modified_observed_data["LD_obs"].shape[0]
            self.model.set_dim("contigs", n__i, coord_values=np.arange(n__i))
            pm.set_data(modified_observed_data)
            idata = pm.sample(
                **self.sample_kwargs,
            )
        return idata

    def log_likelihood__i(self, excluded_obs, idata__i):
        from scipy.stats import multivariate_normal
        import xarray as xr

        post = idata__i.posterior
        mu = np.asarray(post["LD_mu"])  # (chains, draws, dims)
        sigma2 = np.asarray(post["LD_sigma2"])  # (chains, draws, dims)
        obs = np.asarray(excluded_obs["LD_obs"])[0]  # (dims,)
        chains, draws, dims = mu.shape
        ll = np.empty((chains, draws))

        for c in range(chains):
            for d in range(draws):
                cov = np.diag(sigma2[c, d])  # diagonal covariance
                ll[c, d] = multivariate_normal.logpdf(obs, mean=mu[c, d], cov=cov)
        return xr.DataArray(
            ll,
            coords={"chain": post.coords["chain"], "draw": post.coords["draw"]},
            dims=("chain", "draw"),
        )

    def sel_observations(self, idx):
        assert self.idata_orig is not None
        mean_obs = self.idata_orig["constant_data"]["LD_obs"]
        mask = np.isin(np.arange(mean_obs.shape[0]), idx)
        data__i = {"LD_obs": mean_obs[~mask, :]}
        data_ex = {"LD_obs": mean_obs[idx, :]}
        return data__i, data_ex
