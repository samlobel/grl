from .mdp import MDP, AbstractMDP

# import numpy as np
import jax.numpy as np
from jax.config import config

config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

def memory_cross_product(amdp, T_mem):
    """
    Returns AMDP resulting from cross product of the underlying MDP with given memory function

    :param amdp:  AMDP
    :param T_mem: memory transition function
    """
    T = amdp.T
    R = amdp.R
    phi = amdp.phi
    n_states = T.shape[-1]
    n_states_m = T_mem.shape[-1]
    n_states_x = n_states * n_states_m # cross (x) product MDP
    T_x = np.zeros((T.shape[0], n_states_x, n_states_x))
    R_x = np.zeros((R.shape[0], n_states_x, n_states_x))
    phi_x = np.zeros((n_states_x, phi.shape[-1] * n_states_m))

    # Rewards only depend on MDP (not memory function)
    R_x = R.repeat(n_states_m, axis=1).repeat(n_states_m, axis=2)

    # T_mem_phi is like T_pi
    # It is SMxM
    T_mem_phi = np.tensordot(phi, T_mem, axes=1)

    # Outer product that compacts the 2 i
    T_x = np.einsum('ijk,lim->lijmk', T_mem_phi, T).reshape(T.shape[0], n_states_x, n_states_x)

    # The new obs_x are the original obs times memory states
    # E.g. obs={r,b} and mem={0,1} -> obs_x={r0,r1,b0,b1}
    phi_x = np.kron(phi, np.eye(n_states_m))

    mdp_x = MDP(T_x, R_x, amdp.gamma)
    # Assuming memory starts with all 0s
    p0_x = np.zeros(n_states_x)
    # p0_x[::n_states_m] = amdp.p0
    p0_x = p0_x.at[::n_states_m].set(amdp.p0)
    return AbstractMDP(mdp_x, phi_x, p0=p0_x)
