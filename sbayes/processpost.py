import math
import numpy as np

def compute_dic(lh, burn_in):
    """This function computes the deviance information criterion
    (see for example Celeux et al. 2006) using the posterior mode as a point estimate
    Args:
        lh: (dict): log-likelihood of samples in hte posterior
        burn_in(float): percentage of samples, which are discarded as burn-in
        """
    end_bi = math.ceil(len(lh) * burn_in)
    lh = lh[end_bi:]

    # max(ll) likelihood evaluated at the posterior mode
    d_phi_pm = -2 * np.max(lh)
    mean_d_phi = -4 * (np.mean(lh))

    dic = mean_d_phi + d_phi_pm

    return dic