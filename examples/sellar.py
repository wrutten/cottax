'''
Sellar (Sellar, Batill & Renaud 1996), declared as Martins & Lambe (2013, Eq. 1) state the MDO
problem: the two disciplines, the objective as its discipline terms and their sum, each design
constraint and the bounds of each design variable a requirement of its own, and one optimiser
whose unknowns are the design variables. Nothing in it chooses an architecture: `declared()`
is the base every preset starts from.

Run as a script, it answers that base under the monolithic architectures and prints each
answer against the literature's optimum:

    JAX_PLATFORMS=cpu python examples/sellar.py

On Sellar the disciplines are explicit, so a coupling has no state: SAND builds IDF's graph,
and AAO is IDF with every coupling copied (`Jacobi`).
'''

import jax
import jax.numpy as jnp

import cottax.assemblies.jax as cx
from cottax.assemblies.jax import flat


# ---------------------------------------------------------------- the disciplines

@flat.function('y1')
def D1(z1, z2, x1, y2):
    return z1**2 + z2 + x1 - 0.2 * y2


@flat.function('y2')
def D2(z1, z2, y1):
    return jnp.sqrt(y1) + z1 + z2


# ---------------------------------------------------------------- the helper functions

@flat.function('f1')
def F1(x1, y1):
    return x1**2 + y1


@flat.function('f2')
def F2(z2, y2):
    return z2 + jnp.exp(-y2)


@flat.function('f')
def F(f1, f2):
    return f1 + f2


@flat.inequality_constraint('g1')
def G1(y1):
    return 1.0 - y1 / 3.16


@flat.inequality_constraint('g2')
def G2(y2):
    return y2 / 24.0 - 1.0


# ---------------------------------------------------------------- the design variables

BOUNDS = {'z1': (-10.0, 10.0), 'z2': (0.0, 10.0), 'x1': (0.0, 10.0)}
'''Each design variable's `(lo, hi)`.'''


def bound(name: str, lo: float, hi: float) -> tuple[cx.RelationalCondition, dict]:
    '''`lo <= x <= hi` as a requirement of its own, so it goes wherever `x` does, and the
    values it reads: a side of zero is no place, any other a place minted at `^lower` or
    `^upper`.'''
    x, given = flat.place(name), {}
    def side(value, purpose):
        if value == 0:
            return None
        at = flat.place(purpose(name))
        given[at] = value
        return at
    low, high = side(lo, cx.lower), side(hi, cx.upper)
    requirement = cx.RelationalCondition(
        (cx.Relation(low, x, cx.Le), cx.Relation(x, high, cx.Le)), (), (),
        tuple(p for p in (low, high) if p is not None))
    return requirement, given


def declared() -> cx.Declared:
    '''Sellar as declared: the models, the helper functions, the requirements, and the
    optimiser at `opt` over `z1`, `z2`, `x1`.'''
    bounds = {name: bound(name, *pair) for name, pair in BOUNDS.items()}
    opt    = flat.optimise(minimise='f', design=tuple(BOUNDS))
    model  = flat.declare(D1, D2, F1, F2, F, G1, G2,
                          {'opt': opt, 'bounds': {name: r for name, (r, _) in bounds.items()}})
    return model.using({at: v for _, given in bounds.values() for at, v in given.items()})


# ---------------------------------------------------------------- the monolithic reference

OPTIMUM = {'f': 3.18339, 'z1': 1.97764, 'z2': 0.0, 'x1': 0.0, 'y1': 3.16, 'y2': 3.75528}
'''The literature's optimum.'''

START = {'z1': 5.0, 'z2': 2.0, 'x1': 1.0, 'y1': 1.0, 'y2': 1.0}
'''One state for every architecture: a copy's start is read at what it copies.'''


def architectures() -> dict[str, cx.GraphOp]:
    '''The monolithic architectures under `opt`.'''
    return {'SAND': cx.flat.sand('opt'),
            'IDF':  cx.flat.idf('opt'),
            'AAO':  cx.flat.idf('opt', cx.Jacobi()),
            'MDF':  cx.flat.mdf('opt')}


def answered(architecture: cx.GraphOp, base: cx.Declared | None = None) -> dict[str, float]:
    '''The base under `architecture`, driven by the shipped solvers and run from `START`.'''
    model = (declared() if base is None else base) + architecture
    model = model + cx.flat.drive(cx.SLSQP(), cx.Picard(), cx.Newton())
    out   = flat.run(model, read_from={cx.Start: START})
    return {name: float(out[name]) for name in OPTIMUM}


def main(tol: float = 1e-4) -> None:
    base = declared()
    print(f"{'':6}" + ''.join(f'{name:>10}' for name in OPTIMUM))
    print(f"{'ref':6}" + ''.join(f'{value:>10.5f}' for value in OPTIMUM.values()))
    for name, architecture in architectures().items():
        answer = answered(architecture, base)
        worst  = max(abs(answer[k] - OPTIMUM[k]) for k in OPTIMUM)
        print(f'{name:6}' + ''.join(f'{value:>10.5f}' for value in answer.values())
              + ('   ok' if worst < tol else f'   off by {worst:.1e}'))


if __name__ == '__main__':
    with jax.enable_x64(True):
        main()
