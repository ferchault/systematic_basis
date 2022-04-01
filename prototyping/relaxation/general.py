#!/usr/bin/env python
# usage: python general.py [Number s primitives] [Number p primitives] ...
import pyscf.gto
import pyscf.scf
import scipy.optimize as sco
import numpy as np
import sys


def even_tempered(l: int, alpha: float, beta: float, N: int):
    return alpha * beta ** np.arange(1, N + 1)


def to_basis(exponents, Ns):
    return [[N, [exponent, 1]] for exponent, N in zip(exponents, Ns)]


def do_mol(basis):
    try:
        mol = pyscf.gto.M(
            atom=f"Be 0 0 -1.226791616343017; Be 0 0 1.226791616343017",
            basis=basis,
            verbose=0,
        )
    except:
        return 0
    calc = pyscf.scf.RHF(mol)
    try:
        e = calc.kernel()
        if not calc.converged:
            return 0
        return e
    except:
        return 0


def args_to_bas(x0, Ns):
    x0 = np.abs(x0)
    nl = int(len(x0) / 2)
    bas = []
    for l, N in zip(range(nl), Ns):
        alpha, beta = x0[l * 2 : l * 2 + 2]
        bas += to_basis(even_tempered(l, alpha, beta, N), [l] * N)
    return bas


def first_stage(x0, Ns):
    return do_mol({"Be": args_to_bas(x0, Ns)})


if __name__ == "__main__":
    CBS = -29.1341759449
    try:
        args = [int(_) for _ in sys.argv[1:]]
        x0 = [400, 0.2] * len(args)
        res = sco.minimize(first_stage, x0, args=(args,))
        print("Basis set")
        print(args_to_bas(res.x, args))
        print("Error to CBS [Ha]")
        print(res.fun - CBS)
    except:
        basis = sys.argv[1]
        print("Contracted", do_mol(basis) - CBS)
        print(
            "Uncontracted",
            do_mol({"Be": pyscf.gto.uncontract(pyscf.gto.basis.load(basis, "Be"))})
            - CBS,
        )
