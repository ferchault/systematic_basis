import dqc
import torch
import xitorch as xt
import xitorch.optimize
import numpy as np
from dqc.utils.datastruct import CGTOBasis
from collections.abc import Iterable


def optimize_fixed_basis(
    nuclear_charges: Iterable[float],
    nuclear_coordinates: Iterable[Iterable[float]],
    primitive_centers: Iterable[Iterable[float]],
    primitive_exponents: Iterable[float],
    net_charge: float,
    spin: int,
):
    basis = []
    charges = []
    coords = []
    for center, exponent in zip(primitive_centers, primitive_exponents):
        alphas = torch.tensor([exponent], dtype=torch.float64)
        coeffs = torch.tensor([1], dtype=torch.float64)
        basis.append(CGTOBasis(0, alphas, coeffs))
        charges.append(0)
        coords.append(center)

    for center, charge in zip(nuclear_coordinates, nuclear_charges):
        coords.append(center)
        charges.append(charge)
        basis.append(CGTOBasis(0, torch.tensor([]), torch.tensor([])))

    bpacker = xt.Packer(basis)
    bparams = bpacker.get_param_tensor()

    def fcn(bparams, bpacker):
        basis = bpacker.construct_from_tensor(bparams)

        m = dqc.Mol((charges, coords), basis=basis, charge=net_charge, spin=spin)
        qc = dqc.HF(m)

        qc = qc.run()
        qc.get_system().get_nuclei_energy = lambda: 0
        ene = qc.energy()
        return ene

    min_bparams = xitorch.optimize.minimize(
        fcn,
        bparams,
        (bpacker,),
        method="gd",
        step=1e-2,
        maxiter=10000,
        verbose=True,
        f_tol=1e-10,
        x_tol=1e-10,
        f_rtol=1e-10,
        x_rtol=1e-10,
    )
    opt_basis = bpacker.construct_from_tensor(min_bparams)

    alphas = [_.alphas.detach().numpy() for _ in opt_basis[: len(primitive_exponents)]]
    return np.array(alphas).flatten(), fcn(min_bparams, bpacker).detach().numpy()


import sys

Z = float(sys.argv[1])
nprimitives = int(sys.argv[2])

guess = 0.1 * 4.1 ** np.arange(0, nprimitives)
# guess = [1.33249899, 0.20152957]
net_charge = Z - 1
alphas, energy = optimize_fixed_basis(
    (Z,), ((0, 0, 0),), [(0, 0, 0)] * nprimitives, guess, net_charge, 1
)
print(f"run-{Z}-{nprimitives}")
print("result-alpha " + " ".join([str(_) for _ in alphas]))
print(f"result-energy {energy}")
