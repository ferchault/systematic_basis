#!/usr/bin/env python

#%%
import pyscf.scf
import pyscf.gto
import functools

# %%
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
        return calc.kernel()
    except:
        return 0


# %%
# mol = [2], [[0,0,0]]
# get_floating_calculator(*mol, )
# mol = pyscf.gto.M(atom=f"He 0 0 0", basis="def2-TZVP", verbose=0)
def sto3g_like(*args):

    a1, a2, a3, a4, a5, a6, a7, a8, a9, c1, c2, c3, c4, c5, c6 = args

    return [
        [0, [a1, c1], [a2, c2], [a3, 1 - c1 - c2]],
        [0, [a4, c3], [a5, c4], [a6, 1 - c3 - c4]],
        [1, [a7, c5], [a8, c6], [a9, 1 - c5 - c6]],
    ]


def sto3g_like(*args):

    a1, a2, a3, a4, a5, a6, a7, a8, a9 = args

    return [
        [0, [a1, 1]],
        [0, [a2, 1]],
        [0, [a3, 1]],
        [0, [a4, 1]],
        [0, [a5, 1]],
        [0, [a6, 1]],
        [1, [a7, 1]],
        [1, [a8, 1]],
        [1, [a9, 1]],
    ]


# %%
import scipy.optimize as sco
import numpy as np


def fun(_):
    return do_mol({"Be": sto3g_like(*_)})


x0 = (30.1, 5.5, 1.5, 1.3, 0.3, 0.1, 1.3, 0.3, 0.1, 0.1, 0.5, -0.1, 0.4, 0.15, 0.6)
x0 = (
    6.66118155e01,
    1.00712830e01,
    2.08456591e00,
    1.63671151e01,
    1.34538182e-01,
    4.86391910e-02,
    5.57089662e-02,
    5.57089414e-02,
    3.32125335e-01,
    6.10807332e-02,
    3.41683130e-01,
    2.50465855e-02,
    7.18503539e-01,
    -2.63337289e-01,
    8.85350521e-01,
)
if __name__ == "__main__":
    bounds = [(0.001, 100)] * 9 + [(0, 1)] * 6
    print("START", do_mol("STO-3G"))

    def callback(xk, convergence=None):
        print(do_mol(sto3g_like(*xk)))

    # sco.differential_evolution(fun, bounds=bounds, callback=callback, workers=-1)

    result = sco.minimize(fun, x0=x0[:9], callback=callback)
    print(result)
    # result = sco.minimize(
    #     fun, x0=x0 + np.random.random(size=len(x0)) * 0.2, callback=callback
    # )
    # print(result)
    # sco.minimize: -28.923121551758687
    # sco.DE: -28.85997709582355

# %%
