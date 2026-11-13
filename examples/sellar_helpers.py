'''
Sellar (Sellar, Batill & Renaud 1996) and what its architectures share: the base every plan
starts from, the names the plans write, the bodies of the functions they insert, a
finite-difference Jacobian for a system subproblem over nested constrained solves, and running
a plan against the literature's optimum. `sellar_plans.py` holds the plans; `sellar.ipynb` runs
them.
'''

import dataclasses

import jax.numpy as jnp

import cottax.assemblies.jax as cx
from cottax.assemblies.jax import flat
from cottax.consumers.visualization.interface_types import Around
from cottax.consumers.visualization.latex import LatexFormatter
from cottax.consumers.visualization.latex.xdsm import LATIN_MODERN
from cottax.core import Function, Mint
from cottax.core.realisation import ImplementedFunction


# ---------------------------------------------------------------- the base
#
# Sellar declared as Martins & Lambe (2013, Eq. 1) state the MDO problem: the two disciplines;
# the objective as its discipline terms and their sum; each design constraint and the bounds
# of each design variable a requirement of its own; one optimiser whose unknowns are the
# design variables. Nothing in it chooses an architecture, so every plan, monolithic or
# distributed, starts from `sellar()`. `D2` takes `√|y1|`, Sellar's form in GEMSEO and
# OpenMDAO, defined for every target `ŷ1` an optimiser may try.

@flat.function('y1')
def D1(z1, z2, x1, y2):
    return z1**2 + z2 + x1 - 0.2 * y2


@flat.function('y2')
def D2(z1, z2, y1):
    return jnp.sqrt(jnp.abs(y1)) + z1 + z2


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


BOUNDS = {'z1': (-10.0, 10.0), 'z2': (0.0, 10.0), 'x1': (0.0, 10.0)}
'''Each design variable's `(lo, hi)`.'''


# WORKAROUND(FR-8): a constraint over a place nothing wraps, written by hand with its values.
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


def sellar() -> cx.Declared:
    '''Sellar as declared: the models, the helper functions, the requirements, and the
    optimiser at `opt` over `z1`, `z2`, `x1`.'''
    bounds = {name: bound(name, *pair) for name, pair in BOUNDS.items()}
    opt    = flat.optimise(minimise='f', design=tuple(BOUNDS))
    model  = flat.declare(D1, D2, F1, F2, F, G1, G2,
                          {'opt': opt, 'bounds': {name: r for name, (r, _) in bounds.items()}})
    return model.using({at: v for _, given in bounds.values() for at, v in given.items()})


OPTIMUM = {'f': 3.18339, 'z1': 1.97764, 'z2': 0.0, 'x1': 0.0, 'y1': 3.16, 'y2': 3.75528}
'''The literature's optimum.'''

LOCAL = {'f': 4.13076, 'z1': -1.71714, 'z2': 0.13846, 'x1': 0.11276, 'y1': 3.16, 'y2': 0.19895}
'''Sellar's other local minimum, where MDF stays when started there.'''

START = {'z1': 5.0, 'z2': 2.0, 'x1': 1.0, 'y1': 1.0, 'y2': 1.0}
'''One state every plan starts from: a copy's start is read at what it copies.'''


# ---------------------------------------------------------------- the names the plans write
#
# Sellar's places and nodes, as the base declares them, and every name an architecture
# derives. A derived name is a `Mint`, a purpose by its class, worn by the name it is derived
# from. cottax ships `Copy`, one copy per variable; a subproblem's own copy of a shared
# variable needs one per subproblem, so each subproblem has a copy mint here.
#
# WORKAROUND(FR-5): `Copy1` and `Copy2`, written by hand for Sellar's two subproblems.

class Copy1(Mint):
    '''Subproblem 1's own copy of a shared variable.'''

class Copy2(Mint):
    '''Subproblem 2's own copy of a shared variable.'''

class Response(Mint):
    '''What a discipline computes for a coupling the optimiser owns, AAO's `Y_i`.'''

class Inconsistency(Mint):
    '''A subproblem's squared inconsistency with the system subproblem, CO's `J_i`.'''

class Penalty(Mint):
    '''A weighted inconsistency, IPD's, EPD's and ATC's `Φ_i`.'''

class SubproblemObjective(Mint):
    '''What a problem minimises, where it is not the declared objective.'''

class Weight(Mint):
    '''A weight: a penalty's, or BLISS-2000's on a discipline's response.'''

class Budget(Mint):
    '''The budget the system subproblem gives a discipline's objective term, QSD's `b_i`.'''

class Margin(Mint):
    '''A subproblem's constraint margin, QSD's `s_i`.'''

class Loosened(Mint):
    '''A constraint loosened by the margin, QSD's `c_i + s_i` and `f_i − b_i + s_i`.'''

class ElasticPlus(Mint):
    '''The positive part of an elastic consistency, EPD's.'''

class ElasticMinus(Mint):
    '''The negative part of an elastic consistency, EPD's.'''

class ElasticResidual(Mint):
    '''A consistency less its elastic split, `(a − b) − (plus − minus)`, held at zero (EPD).'''

class Barrier(Mint):
    '''The barrier on the elastic variables, lowered in EPD's outer loop.'''


place = flat.place
z1, z2, x1, y1, y2, f1, f2, f, g1, g2 = map(place, ('z1', 'z2', 'x1', 'y1', 'y2',
                                                    'f1', 'f2', 'f', 'g1', 'g2'))
D1_NODE, D2_NODE, F1_NODE, F2_NODE, F_NODE = map(flat.node, ('D1', 'D2', 'F1', 'F2', 'F'))
G1_NODE, G2_NODE = flat.node(cx.require('G1')), flat.node(cx.require('G2'))
BOUND_NODES = {k: flat.node(cx.N['bounds'][k]) for k in BOUNDS}
SYSTEM = flat.node('opt')
'''The declared optimiser: a monolithic architecture's one optimiser, a distributed one's
system subproblem.'''
SUBPROBLEM_1, SUBPROBLEM_2 = flat.node('subproblem_1'), flat.node('subproblem_2')

z1_copy1, z2_copy1 = Copy1()(z1), Copy1()(z2)
z1_copy2, z2_copy2 = Copy2()(z1), Copy2()(z2)
y1_hat, y2_hat, x1_hat = cx.Copy()(y1), cx.Copy()(y2), cx.Copy()(x1)
weight, barrier = Weight()(place('w')), Barrier()(place('mu'))

CONSISTENCY = {'D1': ((z1_copy1, z1), (z2_copy1, z2), (y1_hat, y1)),
               'D2': ((z1_copy2, z1), (z2_copy2, z2), (y2_hat, y2))}
'''Each subproblem's consistency with the system subproblem: its copies of `z`, and the
system subproblem's target against its response.'''


# ---------------------------------------------------------------- how the drawings write names
#
# The drawings set names in LaTeX in Martins & Lambe's notation, as the graph paper's figures
# do: a declared name as it stands (`z1` is `z_1`), a copy with a hat over its letter alone
# (`\hat{y}_1`), a subproblem's own copy with its index (`\hat{z}_1^{(1)}`), a fixed point's
# copy of what the subproblems read with a prime (`z_1^\prime`), a start with `(0)`, a bound
# with `L` or `U`, and every name an architecture adds by its symbol. A problem is named by its
# kind (`Opt`, `Opt_i`, `FP`), a requirement by the relation it holds (`g_1 \leq 0`), a
# problem's own objective by its formula, an inserted function by what it computes. Every
# drawing types a condition node's reads in slots (`min`, `≤ 0`, `= 0`), as the paper's
# figures do.

def _copies() -> dict[str, str]:
    '''Every copy and its start, by plain spelling.'''
    names = {}
    for v in ('z1', 'z2', 'x1', 'y1', 'y2'):
        letter, index = v[0], v[1]
        copies = {f'^copy1.{v}': (rf'\hat{{{letter}}}_{index}', '(1)'),
                  f'^copy2.{v}': (rf'\hat{{{letter}}}_{index}', '(2)'),
                  f'^copy.^copy.{v}': (rf'\hat{{{letter}}}_{index}', r'\prime')}
        copies[f'^copy.{v}'] = ((rf'{letter}_{index}', r'\prime') if letter == 'z' else
                                (rf'\hat{{{letter}}}_{index}', ''))
        for spelled, (body, sup) in copies.items():
            names[spelled] = f'{body}^{{{sup}}}' if sup else body
            names[f'^start.{spelled}'] = f'{body}^{{{sup + "," if sup.startswith("(") else sup}(0)}}'
    return names


NAMES = {
    'opt': r'\text{Opt}', 'subproblem_1': r'\text{Opt}_1', 'subproblem_2': r'\text{Opt}_2',
    '^require.G1': r'g_1 \,\leq\, 0', '^require.G2': r'g_2 \,\leq\, 0',
    'bounds.z1': r'z_1^L \,\leq\, z_1 \,\leq\, z_1^U',
    'bounds.z2': r'0 \,\leq\, z_2 \,\leq\, z_2^U',
    'bounds.x1': r'0 \,\leq\, x_1 \,\leq\, x_1^U',
    'mda': r'\text{FP}', '^closure.D1': r'\text{FP}', 'top': r'\text{FP}',
    'inconsistency_1': 'J_1', 'inconsistency_2': 'J_2',
    'penalty_1': r'\Phi_1', 'penalty_2': r'\Phi_2', 'penalty_sum': r'\Sigma\Phi',
    'penalty_system': r'\Sigma\Phi',
    'margins_1': 'S_1', 'budget_sum': r'f_2 + b_1', 'weighted_1': r'w_1 y_1',
    '^response.y1': 'Y_1', '^response.y2': 'Y_2',
    '^inconsistency.D1': 'J_1', '^inconsistency.D2': 'J_2',
    '^penalty.D1': r'\Phi_1', '^penalty.D2': r'\Phi_2',
    '^subproblemobjective.opt': r'\Sigma\Phi',
    '^subproblemobjective.subproblem_1': r'f_1 + \Phi_1',
    '^subproblemobjective.subproblem_2': r'f_2 + \Phi_2',
    '^weight.w': 'w', '^weight.D1': 'w_1', '^barrier.mu': r'\mu',
    '^budget.D1': 'b_1', '^margin.D1': 's_1',
    '^loosened.g1': r'\bar{g}_1', '^loosened.f1': r'\bar{f}_1', '^loosened.x1': r'\bar{x}_1',
    **{f'^{part}.D{i}_{k}': f'{symbol}_{{{i},{k + 1}}}'
       for part, symbol in (('elasticplus', 'p'), ('elasticminus', 'n'),
                            ('elasticresidual', 'r'))
       for i in (1, 2) for k in range(3)},
    **_copies(),
}
'''Each name drawn outright, by its plain spelling. A plan whose own objectives differ says so
in its `Run.names`.'''


def formatter(names: dict[str, str] | None = None) -> LatexFormatter:
    '''The LaTeX formatter of every drawing: `NAMES` and a plan's own `names` outright (every
    copy among them, so its hat sits over the letter, not the subscript), and the notation of
    each other purpose a drawing meets over the name it stands for.'''
    return LatexFormatter({**NAMES, **(names or {})}, notation={
        type(cx.start.mint): Around(r'{}^{{(0)}}'),
        type(cx.lower.mint): Around(r'{}^L'),
        type(cx.upper.mint): Around(r'{}^U'),
    })


# ---------------------------------------------------------------- what the inserted functions compute
#
# One body per function block of Martins & Lambe's figures ("Discipline i and Penalty
# Functions", "System and Penalty Functions"), so a block is one node with several outputs, not
# a chain of arithmetic. A node calls its body with one argument per read, in order.
#
# WORKAROUND(FR-10): plain bodies, where a preset would insert formulas.

def squared_distance(*pairs):
    '''`Σ (a − b)²` over `a1, b1, a2, b2, …`: CO's inconsistency `J_i`.'''
    return sum((a - b) ** 2 for a, b in zip(pairs[::2], pairs[1::2]))


def penalty(w, *pairs):
    '''`w · Σ (a − b)²`: a quadratic penalty, ATC's system subproblem's over every
    discipline.'''
    return w * squared_distance(*pairs)


def penalty_and_objective(w, term, *pairs):
    '''A subproblem's quadratic penalty `Φ` and its objective `term + Φ` (IPD, ATC).'''
    phi = penalty(w, *pairs)
    return phi, term + phi


def l1_penalty_with_barrier(w, mu, term, *quads):
    '''A subproblem's L1 penalty with a barrier on its elastic variables,
    `Φ = w · Σ (plus + minus) − mu · Σ (log plus + log minus)`, its objective `term + Φ`, and
    each consistency less its elastic split, `(a − b) − (plus − minus)`, over
    `a, b, plus, minus, …` (EPD). SLSQP is no interior-point method and may clip an elastic
    variable onto its bound at zero, so the logarithm is taken of at least `1e-300`.'''
    a, b, plus, minus = (quads[k::4] for k in range(4))
    # WORKAROUND(FR-11): the clip, for an active-set method on a barrier.
    log = lambda v: jnp.log(jnp.maximum(v, 1e-300))
    phi = (w * sum(p + m for p, m in zip(plus, minus))
           - mu * sum(log(p) + log(m) for p, m in zip(plus, minus)))
    return (phi, term + phi,
            *((ai - bi) - (p - m) for ai, bi, p, m in zip(a, b, plus, minus)))


def qsd_margins(g, term, budget, x, s):
    '''A subproblem's constraint, budget and upper bound loosened by its margin, and the
    margin's negative, which it minimises: `g + s`, `term − budget + s`, `x + s`, `−s` (QSD).'''
    return g + s, term - budget + s, x + s, -s


def add(a, b):
    '''`a + b`.'''
    return a + b


def multiply(a, b):
    '''`a · b`.'''
    return a * b


def function_node(body, reads, owns):
    '''A function node's definition (not a graph operation): `body` over `reads`, owning
    `owns`.'''
    return ImplementedFunction(Function(tuple(reads), tuple(owns)), body)


# ---------------------------------------------------------------- derivatives through a nested solve
#
# A system subproblem whose conditions are read through a nested constrained subproblem needs
# their post-optimal derivatives, which no shipped driver gives yet. Until one does, its SLSQP
# takes this forward-difference Jacobian. SLSQP asks it under `jit`, so it is written in
# `jax.numpy`; the nested solves run on the host through their callbacks.
#
# WORKAROUND(FR-6): this class, until cottax ships it in `jax_solvers`.

@dataclasses.dataclass(frozen=True)
class FiniteDifferences:
    '''A Jacobian by forward differences of `step`, for `ScipyMinimize(jacobian=...)`: a
    function of `g`, as `jax.jacfwd`, with `has_aux` as it takes it.'''

    step : float = 1e-6

    def __call__(self, g, has_aux=False):
        value = (lambda x: g(x)[0]) if has_aux else g
        def jacobian(x):
            base = value(x)
            jac = jnp.stack([(value(x + self.step * e) - base) / self.step
                             for e in jnp.eye(x.size, dtype=x.dtype)], axis=-1)
            return (jac, g(x)[1]) if has_aux else jac
        return jacobian


# ---------------------------------------------------------------- running a plan
#
# A plan is driven by the shipped solvers: SLSQP for every optimiser (the system subproblem over
# nested constrained subproblems by finite differences), Picard for a fixed point, Newton for a
# root find, each converged tightly enough for those finite differences; then run from `START`
# and whatever else a plan's own unknowns and inputs need. A penalty method runs Martins &
# Lambe's outer loop (Algorithms 7, 8): once per stage of its weight (or barrier), each run
# warm-started from the last one's answer. The loop is the example's, not the graph's: a nested
# solve cannot yet start from its previous answer. A nested subproblem that does not converge
# either stops the run (`strict`) or returns what it has: a failure cannot yet reach the system
# subproblem as a failed evaluation.
#
# WORKAROUND(FR-12): `run_staged`'s loop over the stages, outside the graph.
# WORKAROUND(FR-1): its warm start, each run from the last one's answer.
# WORKAROUND(FR-3): `strict`, the choice between the two; every plan stops the run.

def with_solvers(model: cx.Declared, differences: bool = False, strict: bool = True,
                 system_strict: bool = True, system_tol: float = 1e-7) -> cx.Declared:
    '''`model` with every problem answered by the shipped solvers, and with `differences` the
    system subproblem by SLSQP over finite differences.'''
    if differences:
        model = (model + flat.assign(SYSTEM, cx.ConstrainedMinimisation(steps=False))
                 + flat.implement({SYSTEM: cx.ScipyMinimize(jacobian=FiniteDifferences(),
                                                            tol=system_tol, strict=system_strict)}))
    tight = dict(rtol=1e-12, atol=1e-12, max_steps=1000)
    return model + flat.drive(cx.ScipyMinimize(strict=strict),
                              cx.Picard(**tight), cx.Newton(**tight))


def run_once(model: cx.Declared, state: dict, starts: dict, inputs: dict, **solvers):
    '''`model` with its solvers, run from `state`, and `starts` and `inputs` by name: every
    value.'''
    values = {}
    if starts:
        values[cx.Start] = starts
    if inputs:
        values[cx.Inputs] = inputs
    return flat.run(with_solvers(model, **solvers), values, read_from={cx.Start: state})


def run_staged(model: cx.Declared, *, starts=None, inputs=None, stages=(),
               **solvers) -> dict[str, float]:
    '''`model` run from `START`, `starts` and `inputs` by name, once per stage of `stages`
    (each stage's inputs added to `inputs`, each run warm-started from the last one's answer);
    the answer at each of the optimum's names, as the run computes it there. A plan that takes
    the objective as terms computes no `f`, so it is reported as `f1 + f2`; in CO, `f` is
    computed at the system subproblem's targets and `x1`, `y` are the subproblems'.'''
    state, starts = dict(START), dict(starts or {})
    for stage in stages or ({},):
        out = run_once(model, state, starts, {**(inputs or {}), **stage}, **solvers)
        state  = {name: float(out[name]) for name in START}
        starts = {u: float(out[u]) for u in starts}
    answer = {name: float(out[name]) for name in OPTIMUM if name != 'f'}
    terms_only = F_NODE not in model.graph
    answer['f'] = float(out['f1']) + float(out['f2']) if terms_only else float(out['f'])
    return {name: answer[name] for name in OPTIMUM}


def compare(answer: dict[str, float], tol: float = 1e-4) -> str:
    '''The optimum, `answer`'s error against it, and whether every error is within `tol`.'''
    error = {k: answer[k] - OPTIMUM[k] for k in OPTIMUM}
    worst = max(abs(e) for e in error.values())
    head  = ''.join(f'{k:>11}' for k in OPTIMUM)
    ref   = ''.join(f'{v:>11.5f}' for v in OPTIMUM.values())
    row   = ''.join(f'{e:>11.1e}' for e in error.values())
    return (f'{"":8}{head}\n{"optimum":8}{ref}\n{"error":8}{row}\n'
            + (f'within {tol:g}' if worst < tol else f'off by {worst:.1e} (tolerance {tol:g})'))


XDSM_SETTINGS = dict(blocks=True, condition_slots=True, stack_inputs=True, box_scale=0.8,
                     var_box_scale=0.5, font=LATIN_MODERN)
'''The graph paper's settings for an XDSM.'''


def xdsm(model: cx.Declared, names: dict[str, str] | None = None, **solvers) -> str:
    '''`model`'s XDSM, with its solvers, so each optimiser's start comes in from Coor as in
    Martins & Lambe's figures.'''
    return cx.render_xdsm_svg(with_solvers(model, **solvers), formatter=formatter(names),
                              **XDSM_SETTINGS)


def declared_xdsm(model: cx.Declared) -> str:
    '''The declared problem's XDSM, as declared: no solver, so the optimiser is drawn tinted
    and each requirement hollow, named by the relation it holds (so without slots, which would
    repeat it), and the diagonal in the order written.'''
    return cx.render_xdsm_svg(model, formatter=formatter(), **{
        **XDSM_SETTINGS, 'condition_slots': False, 'problem_first': False})


# ---------------------------------------------------------------- the overview
#
# Each architecture's outcome, gathered as the notebook runs them: its answer, or why it failed,
# in one table.

def failure(error: Exception) -> str:
    '''Why a run failed, in one line: the solver's own message and the node it was raised at.'''
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    said  = [line for line in lines if 'converge' in line] or lines[-1:] or [type(error).__name__]
    at    = [line for line in lines if line.startswith('raised at')]
    return said[-1].split('RuntimeError: ')[-1] + (f' ({at[-1]})' if at else '')


def overview(results: dict) -> str:
    '''A Markdown table of `results`, each `(answer, tolerance, note)`, or `(None, why it
    failed, note)`.'''
    rows = ['| architecture | result | largest error | error in `f` | tolerance | note |',
            '|---|---|---|---|---|---|']
    for name, (answer, said, note) in results.items():
        if answer is None:
            rows.append(f'| {name} | failed: {said} | | | | {note} |')
            continue
        error = {k: answer[k] - OPTIMUM[k] for k in OPTIMUM}
        worst = max(abs(e) for e in error.values())
        local = max(abs(answer[k] - LOCAL[k]) for k in LOCAL) < said
        result = ('within tolerance' if worst < said else
                  'the other local minimum, f = 4.131' if local else 'off')
        rows.append(f'| {name} | {result} | {worst:.1e} | {error["f"]:+.1e} | {said:g} '
                    f'| {note} |')
    return '\n'.join(rows)
