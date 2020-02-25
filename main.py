"""Perform all analyses.

"""
from os import makedirs
from os.path import exists
from time import time

from models import fit_model, modela, modelb
from figures import fig12, fig3, fig4
from tables import table2, table3


def main():

    print("performing all analyses for the 'perceptual anchors' paper")
    started = time()

    details = [
        ("modela", modela, True), ("modelb", modelb, False)
    ]

    if not exists("manuscript"):
        makedirs("manuscript")

    for name, func, eps in details[:]:

        print("fitting or loading model ... ", end="")
        data = fit_model(name, func)
        print(f"done in {time() - started:.2f} s")

        print("creating figs 1 and 2 ... ", end="")
        fig12(data, name, eps)
        print(f"done in {time() - started:.2f} s")

        print("creating fig 3 ... ", end="")
        fig3(data, name, eps)
        print(f"done in {time() - started:.2f} s")

        print("creating fig 4 ... ", end="")
        fig4(name, func, eps)
        print(f"done in {time() - started:.2f} s")

        print("creating table 2 ... ", end="")
        table2(name, func, eps)
        print(f"done in {time() - started:.2f} s")

        print("creating table 3 ... ", end="")
        table3(name, func, eps)
        print(f"done in {time() - started:.2f} s")


if __name__ == '__main__':
    main()
