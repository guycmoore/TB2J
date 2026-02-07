##
# Multipole expansion calculations
##

import sympy as sp
from sympy.physics.wigner import wigner_3j
from sympy import sqrt, factorial, simplify, I

import numpy as np
from scipy.special import sph_harm_y

def n_lk(l, k):
    """
    Calculate normalization constant n_{lk} from Eq. (5).
    n_{lk} = (2l)! / sqrt[(2l-k)!(2l+k+1)!]
    """
    numerator = factorial(2*l)
    denominator = sqrt(factorial(2*l - k) * factorial(2*l + k + 1))
    return numerator / denominator

def mu_matrix_complex(l, k, t, normalize=True):
    """
    Construct the complex μ̂^{kt} matrix for given (l,k,t).
    This implements Eq. (3) and (5) from the paper.
    Returns a (2l+1)x(2l+1) sympy.Matrix.
    """
    dim = 2*l + 1
    mu = sp.zeros(dim, dim)
    n = n_lk(l, k)
    
    # Matrix indices correspond to m values from l to -l
    for i in range(dim):
        m = l - i  # m ranges from l down to -l
        for j in range(dim):
            mp = l - j  # m' ranges from l down to -l
            
            # From Eq. (3) and (5): μ̂^kt_mm' = (-1)^(l-m+k) * (l k l; -m t m') * n_lk^(-1)
            phase = (-1)**(l - m + k)
            w3j = wigner_3j(l, k, l, -m, t, mp)
            mu[i, j] = phase * w3j / n
            
            # Normalize μ̂
            if normalize:
                ck = simplify((2*k + 1) * n**2)
                mu[i, j] = mu[i, j] * sqrt(ck)
            
    return simplify(mu)

def mu_matrix_real(l, k, t, normalize=True):
    """
    Construct the real μ^{kt} matrix using transformation from Appendix A Eq. (A1).
    """
    if t < 0:
        # μ^{k,t<0} = (μ̂^{k,|t|} - (-1)^t μ̂^{k,-|t|}) * i/√2
        mu_pos = mu_matrix_complex(l, k, abs(t), normalize)
        mu_neg = mu_matrix_complex(l, k, -abs(t), normalize)
        mu = (mu_pos - (-1)**abs(t) * mu_neg) * I / sqrt(2)
        # mu = (mu_pos - (-1)**abs(t) * mu_neg) / sqrt(2)
    elif t == 0:
        # μ^{k,0} = μ̂^{k,0}
        mu = mu_matrix_complex(l, k, 0, normalize)
    else:  # t > 0
        # μ^{k,t>0} = (μ̂^{k,-t} + (-1)^t μ̂^{k,t}) / √2
        mu_neg = mu_matrix_complex(l, k, -t, normalize)
        mu_pos = mu_matrix_complex(l, k, t, normalize)
        mu = (mu_neg + (-1)**t * mu_pos) / sqrt(2)
    
    return simplify(mu)

def verify_completeness_relation(l, k_max=None, normalize=True):
    """
    Verify the first completeness relation from Eq. (A2):
    Σ_{mm'} μ^{kt}_{mm'} μ^{k't'}_{m'm} = δ_{kk'} δ_{tt'} / c_k
    """
    if k_max is None:
        k_max = min(2*l, 2)  # Limit for computational efficiency
    
    results_k, results_kt = {}, {}
    
    for k in range(0, k_max + 1):
        results_kt[k] = {}
        for t in range(-k, k + 1):
            mu1 = mu_matrix_real(l, k, t, normalize)
            # mu1 = mu_matrix_complex(l, k, t, normalize)
            
            # Compute c_k from self-product
            sum_self = 0
            dim = 2*l + 1
            for m in range(dim):
                for mp in range(dim):
                    sum_self += mu1[m, mp] * mu1[mp, m]
            
            norm_sq = simplify(sum_self)
            
            # Store norm squared value
            if k not in results_k:
                results_k[k] = {'norm_sq': norm_sq, 'verified': True}
            if t not in results_kt[k]:
                results_kt[k][t] = {'norm_sq': norm_sq}
            
            # Verify orthogonality with other (k',t')
            for kp in range(0, k_max + 1):
                for tp in range(-kp, kp + 1):
                    if k == kp and t == tp:
                        continue  # Skip self-comparison
                    
                    mu2 = mu_matrix_real(l, kp, tp, normalize)
                    # mu2 = mu_matrix_complex(l, kp, tp, normalize)
                    
                    # Compute Σ_{mm'} μ1_{mm'} μ2_{m'm}
                    sum_prod = 0
                    for m in range(dim):
                        for mp in range(dim):
                            sum_prod += mu1[m, mp] * mu2[mp, m]
                    
                    sum_prod = simplify(sum_prod)
                    
                    if sum_prod != 0:
                        results_k[k]['verified'] = False
                        print(f"Non-orthogonal: (k={k},t={t}) and (k'={kp},t'={tp}), sum={sum_prod}")
    
    return results_k, results_kt

def compute_ck_values(l, k_max=None):
    """
    Compute c_k values according to the paper's formula:
    c_k = (2k+1) * n_{lk}^2
    """
    if k_max is None:
        k_max = 2*l
    
    ck_dict = {}
    for k in range(0, k_max + 1):
        n = n_lk(l, k)
        ck_dict[k] = simplify((2*k + 1) * n**2)
    
    return ck_dict

##

def sph_harm_real(l, m, theta, phi):
    """
    Real (tesseral) spherical harmonic evaluated on grids theta_g, phi_g.
    Uses the standard convention:
      Y_real(l,0) = Y_l^0
      Y_real(l,m>0) = sqrt(2)*(-1)^{|m|} * Re[Y_l^m]
      Y_real(l,m<0) = sqrt(2)*(-1)^{|m|} * Im[Y_l^{|m|}]
    """
    if m == 0:
        f = sph_harm_y(l, abs(m), theta, phi)
    elif m > 0:
        f = ((-1)**abs(m)) * np.sqrt(2.0) * sph_harm_y(l, abs(m), theta, phi).real
    else:  # m < 0
        f = ((-1)**abs(m)) * np.sqrt(2.0) * sph_harm_y(l, abs(m), theta, phi).imag
    
    # print("Re, Im = ", np.sum(np.abs(f.real)), np.sum(np.abs(f.imag)))
    
    f = f.real
    
    return f

def evaluate_angular_density(rho_matrix, l, thetas, phis):
    """_summary_

    Args:
        rho_matrix (_type_): _description_
        l (_type_): _description_
        thetas (_type_): _description_
        phis (_type_): _description_

    Returns:
        _type_: _description_
    """

    rho_l = np.zeros_like(thetas, np.complex64)
    for i,m in enumerate(range(-l,l+1)):
        for j,mp in enumerate(range(-l,l+1)):

            sharm_y_lm  = sph_harm_real(l,  m, thetas, phis)
            sharm_y_lmp = sph_harm_real(l, mp, thetas, phis)

            rho_l += rho_matrix[i,j] * sharm_y_lm * sharm_y_lmp

    return rho_l

##

def spherical_to_cartesian(r, theta, phi):
    """
    Convert spherical coordinates to Cartesian coordinates.
    
    Parameters:
        r (float or np.ndarray): Radius.
        theta (float or np.ndarray): Polar angle (radians).
        phi (float or np.ndarray): Azimuthal angle (radians).
        
    Returns:
        tuple: Cartesian coordinates (x, y, z).
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def cartesian_to_spherical(x, y, z, rtol=1.0e-12):
    """
    Convert Cartesian coordinates to spherical coordinates.
    
    Parameters:
        x (float or np.ndarray): x-coordinate.
        y (float or np.ndarray): y-coordinate.
        z (float or np.ndarray): z-coordinate.
        
    Returns:
        tuple: Spherical coordinates (r, theta, phi).
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    r += rtol if r < rtol else 0 # Add small number to avoid division by zero
    theta = np.arccos(z / r)  # Add small number to avoid division by zero
    phi = np.arctan2(y, x)
    return r, theta, phi

def prep_for_cartesian_plot(func, thetas, phis, rescale=True):
    """
    Prepare data for Cartesian plot by converting spherical coordinates to Cartesian.

    Args:
        func (_type_): _description_
        thetas (_type_): _description_
        phis (_type_): _description_
        rescale (bool, optional): _description_. Defaults to True.
    """

    xp, yp, zp = spherical_to_cartesian(np.real(func), thetas, phis)
    xl, yl, zl = np.max(xp)-np.min(xp), np.max(yp)-np.min(yp), np.max(zp)-np.min(zp)
    if rescale:
        xl_max = np.max([xl, yl, zl])
        xl, yl, zl = xl/xl_max, yl/xl_max, zl/xl_max

    return xp, yp, zp, (xl, yl, zl)