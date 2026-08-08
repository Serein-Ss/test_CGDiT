import os
from functools import lru_cache
from math import sqrt, pi

import sympy
import torch
import torch.nn as nn
import numpy as np
from torch_scatter import scatter

CWD = os.path.dirname(os.path.abspath(__file__))
# Precomputed Spherical Bessel function roots in a 2D array with dimension [128, 128]. The n-th (0-based index) root of
# order l Spherical Bessel function is the `[l, n]` entry.
SPHERICAL_BESSEL_ROOTS = torch.tensor(np.load(os.path.join(CWD, "sb_roots.npy")), dtype=torch.float32)


def cosine_cutoff(r: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Cosine cutoff function
    Args:
        r (torch.Tensor): radius distance tensor
        cutoff (float): cutoff distance.

    Returns: cosine cutoff functions

    """
    return torch.where(r <= cutoff, 0.5 * (torch.cos(pi * r / cutoff) + 1), 0.0)


def _block_repeat(array, block_size, repeats):
    col_index = torch.arange(array.size()[1])
    indices = []
    start = 0

    for i, b in enumerate(block_size):
        indices.append(torch.tile(col_index[start : start + b], [repeats[i]]).to(array.device))
        start += b
    indices = torch.cat(indices, axis=0)
    return torch.index_select(array, 1, indices)


@lru_cache(maxsize=128)
def _get_lambda_func(max_n, cutoff: float = 5.0):
    r = sympy.symbols("r")
    en = [i**2 * (i + 2) ** 2 / (4 * (i + 1) ** 4 + 1) for i in range(max_n)]

    dn = [1.0]
    for i in range(1, max_n):
        dn_value = 1 - en[i] / dn[-1]
        dn.append(dn_value)

    fnr = [
        (-1) ** i
        * sqrt(2.0)
        * pi
        / cutoff**1.5
        * (i + 1)
        * (i + 2)
        / sympy.sqrt(1.0 * (i + 1) ** 2 + (i + 2) ** 2)
        * (
            sympy.sin(r * (i + 1) * pi / cutoff) / (r * (i + 1) * pi / cutoff)
            + sympy.sin(r * (i + 2) * pi / cutoff) / (r * (i + 2) * pi / cutoff)
        )
        for i in range(max_n)
    ]

    gnr = [fnr[0]]
    for i in range(1, max_n):
        gnr_value = 1 / sympy.sqrt(dn[i]) * (fnr[i] + sympy.sqrt(en[i] / dn[i - 1]) * gnr[-1])
        gnr.append(gnr_value)
    return [sympy.lambdify([r], sympy.simplify(i), torch) for i in gnr]


def _y00(theta, phi):
    r"""Spherical Harmonics with `l=m=0`.

    ..math::
        Y_0^0 = \frac{1}{2} \sqrt{\frac{1}{\pi}}

    Args:
        theta: torch.Tensor, the azimuthal angle
        phi: torch.Tensor, the polar angle

    Returns: `Y_0^0` results
    """
    return 0.5 * torch.ones_like(theta) * sqrt(1.0 / pi)


def combine_sbf_shf(sbf, shf, max_n: int, max_l: int, use_phi: bool):
    """Combine the spherical Bessel function and the spherical Harmonics function.

    For the spherical Bessel function, the column is ordered by
        [n=[0, ..., max_n-1], n=[0, ..., max_n-1], ...], max_l blocks,

    For the spherical Harmonics function, the column is ordered by
        [m=[0], m=[-1, 0, 1], m=[-2, -1, 0, 1, 2], ...] max_l blocks, and each
        block has 2*l + 1
        if use_phi is False, then the columns become
        [m=[0], m=[0], ...] max_l columns

    Args:
        sbf: torch.Tensor spherical bessel function results
        shf: torch.Tensor spherical harmonics function results
        max_n: int, max number of n
        max_l: int, max number of l
        use_phi: whether to use phi
    Returns:
    """
    if sbf.size()[0] == 0:
        return sbf

    if not use_phi:
        repeats_sbf = torch.tensor([1] * max_l * max_n)
        block_size = [1] * max_l
    else:
        # [1, 1, 1, ..., 1, 3, 3, 3, ..., 3, ...]
        repeats_sbf = np.repeat(2 * torch.arange(max_l) + 1, repeats=max_n)  # type:ignore[assignment]
        # tf.repeat(2 * tf.range(max_l) + 1, repeats=max_n)
        block_size = 2 * torch.arange(max_l) + 1  # type: ignore
        # 2 * tf.range(max_l) + 1
    repeats_sbf = repeats_sbf.to(sbf.device)
    expanded_sbf = torch.repeat_interleave(sbf, repeats_sbf, 1)
    expanded_shf = _block_repeat(shf, block_size=block_size, repeats=[max_n] * max_l)
    shape = max_n * max_l
    if use_phi:
        shape *= max_l
    return torch.reshape(expanded_sbf * expanded_shf, [-1, shape])


def polynomial_cutoff(r: torch.Tensor, cutoff: float, exponent: int = 3) -> torch.Tensor:
    """Envelope polynomial function that ensures a smooth cutoff.

    Ensures first and second derivative vanish at cuttoff. As described in:
        https://arxiv.org/abs/2003.03123

    Args:
        r (torch.Tensor): radius distance tensor
        cutoff (float): cutoff distance.
        exponent (int): minimum exponent of the polynomial. Default is 3.
            The polynomial includes terms of order exponent, exponent + 1, exponent + 2.

    Returns: polynomial cutoff function
    """
    coef1 = -(exponent + 1) * (exponent + 2) / 2
    coef2 = exponent * (exponent + 2)
    coef3 = -exponent * (exponent + 1) / 2
    ratio = r / cutoff
    poly_envelope = 1 + coef1 * ratio**exponent + coef2 * ratio ** (exponent + 1) + coef3 * ratio ** (exponent + 2)

    return torch.where(r <= cutoff, poly_envelope, 0.0)


def get_spherical_bessel_roots(n: int) -> torch.Tensor:
    """预计算球贝塞尔函数的根 (用于基函数扩展)"""
    rng = np.arange(1, n + 1)
    roots = rng * np.pi
    return torch.from_numpy(roots).float()


class GaussianExpansion(nn.Module):
    """Gaussian Radial Expansion.

    The bond distance is expanded to a vector of shape [m], where m is the number of Gaussian basis centers.
    """

    def __init__(
        self,
        initial: float = 0.0,
        final: float = 4.0,
        num_centers: int = 20,
        width: None | float = 0.5,
    ):
        """
        Args:
            initial: Location of initial Gaussian basis center.
            final: Location of final Gaussian basis center
            num_centers: Number of Gaussian Basis functions
            width: Width of Gaussian Basis functions.
        """
        super().__init__()
        self.centers = nn.Parameter(torch.linspace(initial, final, num_centers), requires_grad=False)  # type: ignore
        if width is None:
            self.width = 1.0 / torch.diff(self.centers).mean()
        else:
            self.width = width

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.centers = nn.Parameter(self.centers, requires_grad=False)

    def forward(self, bond_dists):
        """Expand distances.

        Args:
            bond_dists :
                Bond (edge) distances between two atoms (nodes)

        Returns:
            A vector of expanded distance with shape [num_centers]
        """
        diff = bond_dists[:, None] - self.centers[None, :]
        return torch.exp(-self.width * (diff**2))


class SphericalBesselFunction(nn.Module):
    """Calculate the spherical Bessel function based on sympy + pytorch implementations."""

    def __init__(self, max_l: int, max_n: int = 5, cutoff: float = 5.0, smooth: bool = False):
        """Args:
        max_l: int, max order (excluding l)
        max_n: int, max number of roots used in each l
        cutoff: float, cutoff radius
        smooth: Whether to smooth the function.
        """
        super().__init__()
        self.max_l = max_l
        self.max_n = max_n
        self.register_buffer("cutoff", torch.tensor(cutoff))
        self.smooth = smooth
        if smooth:
            self.funcs = self._calculate_smooth_symbolic_funcs()
        else:
            self.funcs = self._calculate_symbolic_funcs()

    @lru_cache(maxsize=128)
    def _calculate_symbolic_funcs(self) -> list:
        """Spherical basis functions based on Rayleigh formula. This function
        generates
        symbolic formula.

        Returns: list of symbolic functions
        """
        x = sympy.symbols("x")
        funcs = [sympy.expand_func(sympy.functions.special.bessel.jn(i, x)) for i in range(self.max_l + 1)]
        return [sympy.lambdify(x, func, torch) for func in funcs]

    @lru_cache(maxsize=128)
    def _calculate_smooth_symbolic_funcs(self) -> list:
        return _get_lambda_func(max_n=self.max_n, cutoff=self.cutoff)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Args:
            r: torch.Tensor, distance tensor, 1D.

        Returns:
            torch.Tensor: [n, max_n * max_l] spherical Bessel function results
        """
        if self.smooth:
            return self._call_smooth_sbf(r)
        return self._call_sbf(r)

    def _call_smooth_sbf(self, r):
        results = [i(r) for i in self.funcs]
        return torch.t(torch.stack(results))

    def _call_sbf(self, r):
        r_c = r.clone()
        r_c[r_c > self.cutoff] = self.cutoff
        roots = SPHERICAL_BESSEL_ROOTS[: self.max_l, : self.max_n]
        roots = roots.to(r.device)

        results = []
        factor = torch.tensor(sqrt(2.0 / self.cutoff**3))
        factor = factor.to(r_c.device)
        for i in range(self.max_l):
            root = roots[i].clone()
            func = self.funcs[i]
            func_add1 = self.funcs[i + 1]
            results.append(
                func(r_c[:, None] * root[None, :] / self.cutoff) * factor / torch.abs(func_add1(root[None, :]))
            )
        return torch.cat(results, axis=1)

    @staticmethod
    def rbf_j0(r, cutoff: float = 5.0, max_n: int = 3):
        """Spherical Bessel function of order 0, ensuring the function value
        vanishes at cutoff.

        Args:
            r: torch.Tensor pytorch tensors
            cutoff: float, the cutoff radius
            max_n: int max number of basis

        Returns:
            basis function expansion using first spherical Bessel function
        """
        n = (torch.arange(1, max_n + 1)).type(dtype=torch.float32)[None, :]
        r = r[:, None]
        return sqrt(2.0 / cutoff) * torch.sin(n * pi / cutoff * r) / r


class ExpNormalFunction(nn.Module):
    """Implementation of radial basis function using exponential normal smearing."""

    def __init__(self, cutoff: float = 5.0, num_rbf: int = 50, learnable: bool = True):
        """
        Initialize ExpNormalSmearing.

        Args:
            cutoff (float): The cutoff distance beyond which interactions are considered negligible. Default is 5.0.
            num_rbf (int): The number of radial basis functions (RBF) to use. Default is 50.
            learnable (bool): If True, the means and betas parameters are learnable.
                              If False, they are fixed. Default is True.
        """
        super().__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.learnable = learnable

        self.alpha = 5.0 / cutoff

        means, betas = self._initial_params()
        if learnable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        """Initialize the means and betas parameters."""
        start_value = torch.exp(torch.tensor(-self.cutoff, dtype=torch.float32))
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf)
        return means, betas

    def forward(self, r: torch.Tensor):
        """
        Compute the radial basis function for the input distances.

        Args:
            r (torch.Tensor): Input distances.

        Returns:
            torch.Tensor: Smearing function applied to the input distances.
        """
        r = r.unsqueeze(-1)
        cutoff = torch.as_tensor(self.cutoff).item()  # type: ignore[assignment]
        betas = torch.as_tensor(self.betas)  # type: ignore[assignment]
        means = torch.as_tensor(self.means)  # type: ignore[assignment]
        return cosine_cutoff(r, cutoff) * torch.exp(-betas * (torch.exp(self.alpha * (-r)) - means) ** 2)


class SphericalHarmonicsFunction(nn.Module):
    """Spherical Harmonics function."""

    def __init__(self, max_l: int, use_phi: bool = True):
        """
        Args:
            max_l: int, max l (excluding l)
            use_phi: bool, whether to use the polar angle. If not,
            the function will compute `Y_l^0`.
        """
        super().__init__()
        self.max_l = max_l
        self.use_phi = use_phi
        funcs = []
        theta, phi = sympy.symbols("theta phi")
        for lval in range(self.max_l):
            m_list = range(-lval, lval + 1) if self.use_phi else [0]  # type: ignore
            for m in m_list:
                func = sympy.functions.special.spherical_harmonics.Znm(lval, m, theta, phi).expand(func=True)
                funcs.append(func)
        # replace all theta with cos(theta)
        cos_theta = sympy.symbols("costheta")
        funcs = [i.subs({theta: sympy.acos(cos_theta)}) for i in funcs]
        self.orig_funcs = [sympy.simplify(i).evalf() for i in funcs]
        self.funcs = [sympy.lambdify([cos_theta, phi], i, [{"conjugate": torch.conj}, torch]) for i in self.orig_funcs]
        self.funcs[0] = _y00

    def __call__(self, cos_theta, phi=None):
        """Args:
            cos_theta: Cosine of the azimuthal angle
            phi: torch.Tensor, the polar angle.

        Returns:
            torch.Tensor: [n, m] spherical harmonic results, where n is the number
            of angles. The column is arranged following
            `[Y_0^0, Y_1^{-1}, Y_1^{0}, Y_1^1, Y_2^{-2}, ...]`
        """
        # cos_theta = torch.tensor(cos_theta, dtype=torch.complex64)
        # phi = torch.tensor(phi, dtype=torch.complex64)
        return torch.stack([func(cos_theta, phi) for func in self.funcs], axis=1)
        # results = results.type(dtype=DataType.torch_float)
        # return results

