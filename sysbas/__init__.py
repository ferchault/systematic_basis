from collections.abc import Iterable
import pyscf.gto
import pyscf.scf
import pyscf.qmmm
import numpy as np
from functools import partialmethod


def _add_qmmm(calc, mol, nuclear_coordinates, nuclear_charges):
    """Alters the effective nuclear charges by adding point charges on top of atoms."""
    mf = pyscf.qmmm.mm_charge(calc, nuclear_coordinates, nuclear_charges)

    def energy_nuc(self, charges=None, coords=None):
        return pyscf.gto.mole.energy_nuc(
            nuclear_charges, np.array(nuclear_coordinates) / 0.52917721067
        )

    mf.mol.energy_nuc = energy_nuc.__get__(mf, mf.__class__)

    return mf


def get_floating_calculator(
    nuclear_charges: Iterable[float],
    nuclear_coordinates: Iterable[Iterable[float]],
    primitive_centers: Iterable[Iterable[float]],
    primitive_exponents: Iterable[float],
    net_charge: float,
    spin: int,
) -> pyscf.scf.RHF:
    """Builds a RHF calculator with off-center / floating basis functions.

    Parameters
    ----------
    nuclear_charges : Iterable[float]
        Nuclear charges of the whole system.
    nuclear_coordinates : Iterable[Iterable[float]]
        Coordinates in Angstrom.
    primitive_centers : Iterable[Iterable[float]]
        Coordinates in Angstrom.
    primitive_exponents : Iterable[float]
        exponents in a.u.
    net_charge : float
        Net charge in electrons.
    spin : int
        Multiplicity.

    Returns
    -------
    pyscf.scf.RHF
        A PySCF calculator.
    """

    # define ghost sites for each primitive
    basis = {}
    atomspec = []
    for center, coefficient in zip(primitive_centers, primitive_exponents):
        ghostelement = f"X{len(atomspec)}"
        atomspec.append(f"{ghostelement} {center[0]} {center[1]} {center[2]}")
        basis[ghostelement] = [[0, [coefficient, 1.0]]]

    # build molecule
    mol = pyscf.gto.Mole()
    mol.atom = "; ".join(atomspec)
    mol.basis = basis
    mol.verbose = 0
    mol.spin = spin
    mol.nelectron = sum(nuclear_charges) - net_charge
    mol.build()

    # add external potential
    calc = pyscf.scf.RHF(mol)
    calc = _add_qmmm(calc, mol, nuclear_coordinates, nuclear_charges)
    return calc
