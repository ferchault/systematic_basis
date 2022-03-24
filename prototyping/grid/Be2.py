#!/usr/bin/env python
import pyscf.scf
import pyscf.gto
import numpy as np
import sys
import multiprocessing as mp
import itertools as it


def do_one(args):
    basis, exponent, x, y = args
    internalbasis = {"Be": basis, "X": [[0, [exponent, 1.0]]]}
    mol = pyscf.gto.M(
        atom=f"Be 0 0 -1.226791616343017; Be 0 0 1.226791616343017; X 0 {x} {y}",
        basis=internalbasis,
        verbose=0,
    )
    calc = pyscf.scf.RHF(mol)
    return (exponent, basis, x, y, calc.kernel())


if __name__ == "__main__":
    basis, exponent = sys.argv[1], float(sys.argv[2])
    with mp.Pool(mp.cpu_count()) as p:
        nx = 100
        ny = 100
        xs = np.linspace(0, 10, nx)
        ys = np.linspace(0, 10, ny)
        cases = []
        for coords in it.product(xs, ys):
            cases.append((basis, exponent, *coords))
        cases.append((basis, exponent, 1e10, 1e10))
        results = p.map(do_one, cases)
        for result in results:
            print(*result)
