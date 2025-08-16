import pymc as pm
import numpy as np
import pytensor.tensor as pt
import arviz as az

# -- Arviz wrapper for LOO --
class PyMCModelWrapper(az.PyMCSamplingWrapper):
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
