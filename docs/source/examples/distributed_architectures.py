'''
The MDO architectures of Martins & Lambe (2013) that solve the original problem, each as a plan
on Sellar: a function of the declared base (`sellar_helpers.sellar()`), returning it with the
architecture's ops applied, cottax's own and nothing else. `RUNS` says how each is run and how
close it must come to the literature's optimum; `distributed_architectures_i.ipynb` runs them.

SAND and IDF are cottax's presets under the Jacobi rule, MDF under its default, Gauss-Seidel.
Every other plan is written out against Martins & Lambe's equations and figure. MDOIS is no
plan: it needs no shared design variable, and Sellar shares `z`.

A distributed plan is a recipe of graph operations, applied one at a time and none composed of
others: `Cut`, `Undetermine`, `Determine`, `Rename`, `Delete`, `Insert` of one node, `Combine`,
`Nest`. It has three steps:

1. Remove: the declared graph made acyclic. Every cut first, while the declared optimiser still
   determines the design variables (a cut needs its variable produced); then the optimiser
   undetermined; then what the architecture replaces renamed or deleted.
2. Assemble: each new node inserted, a condition node determining nothing; then each
   requirement combined into the condition node that will own its variables.
3. Add: each problem determined, innermost first, then the problems directly inside it nested
   in it. Problems at one depth go in binding order.

Nodes are bound, and requirements combined, in the order of Martins & Lambe's diagrams: the
system level first, then discipline i = 1, ..., N, its subproblem before its functions. ATC's
top is made in step 3, after the problems it holds are determined: a cut refuses a copy that
already exists, so the top is inserted after its cuts, and the `z` and `ŷ` they move have a
producer only once the system subproblem is determined. The graph paper's Cut, which adds the
closing condition node, is a `Cut` followed by an `Insert` of the closure. The monolithic plans
(the SAND, IDF and MDF presets, and AAO) are outside these steps.

What the plans stand in for, until cottax ships it:

- WORKAROUND(FR-5): a mint per subproblem for its own copies.
- WORKAROUND(FR-10): a plain jax body for each inserted function.
- WORKAROUND(FR-6): forward differences at a system subproblem over nested constrained
  subproblems.
'''

import dataclasses

import cottax.assemblies.jax as cx
from cottax.assemblies.jax import flat
from cottax.core.frozen import OrderedMap
from cottax.core.maths.plan import Combine, Cut, Delete, Insert, Nest, Rename
from cottax.relational import Determine, Undetermine

from sellar_helpers import (
    BOUND_NODES, CONSISTENCY, D1_NODE, D2_NODE, F1_NODE, F2_NODE, F_NODE, G1_NODE, G2_NODE,
    SUBPROBLEM_1, SUBPROBLEM_2, SYSTEM,
    Budget, Copy1, Copy2, ElasticMinus, ElasticPlus, ElasticResidual, Inconsistency, Loosened,
    Margin, Penalty, Response, SubproblemObjective, Weight,
    add, barrier, f, f1, f2, function_node, g1, l1_penalty_with_barrier, multiply, penalty,
    penalty_and_objective, place, qsd_margins, run_staged, squared_distance, weight, x1,
    x1_hat, xdsm, y1, y1_hat, y2, y2_hat, z1, z1_copy1, z1_copy2, z2, z2_copy1, z2_copy2,
)


# ---------------------------------------------------------------- one operation at a time

def insert(name, node):
    '''`node` inserted at `name`: an `Insert` of one binding.'''
    return Insert(OrderedMap(((name, node),)))


def condition(relations=(), objectives=(), parameters=()):
    '''A condition node determining nothing, as a distributed plan inserts one (not a graph
    operation).'''
    return cx.RelationalCondition(relations=tuple(relations), objectives=tuple(objectives),
                                  owns_inside=(), reads_outside=tuple(parameters))


def combine(into, *requirements):
    '''`requirements` combined into the condition node at `into`, which keeps its binding.'''
    return Combine(into, (into, *requirements), cx.RelationalCondition)


# ---------------------------------------------------------------- the monolithic architectures
#
# SAND, IDF and MDF are cottax's presets under the declared optimiser `opt`. Martins & Lambe's
# IDF copies every coupling, so it takes the Jacobi rule; the presets' default, Gauss-Seidel,
# copies only what breaks the cycle. Sellar's disciplines are explicit, so a coupling has no
# state, and SAND builds IDF's graph: the optimiser owns a copy `ŷ_i` and holds `ŷ_i = y_i`
# where Eq. 2 has it own `y_i` and hold `y_i = Y_i`. AAO (Eq. 1) owns both, so it is written
# out: each discipline's output becomes its response `Y_i`, the optimiser owns the couplings
# and their copies, and holds the residuals `y_i = Y_i` and the consistency `ŷ_i = y_i`.

def sand(base): return base + flat.sand('opt', cx.Jacobi())
def idf(base):  return base + flat.idf('opt', cx.Jacobi())
def mdf(base):  return base + flat.mdf('opt')


def aao(base):
    Y1, Y2 = Response()(y1), Response()(y2)
    held = flat.node('consistency')
    return (base
            # each discipline reads the optimiser's targets, and computes its response
            + Cut(y2, (D1_NODE,), cx.Copy()) + Cut(y1, (D2_NODE,), cx.Copy())
            + Rename(D1_NODE, ((y1, Y1),)) + Rename(D2_NODE, ((y2, Y2),))
            # the declared optimiser owns the couplings and their copies as well
            + Undetermine(SYSTEM)
            + Determine(SYSTEM, (z1, z2, x1, y1_hat, y2_hat, y1, y2))
            + Insert(OrderedMap(((held, cx.RelationalCondition(
                relations=(cx.Relation(y1_hat, y1, cx.Eq), cx.Relation(y2_hat, y2, cx.Eq),
                           cx.Relation(y1, Y1, cx.Eq), cx.Relation(y2, Y2, cx.Eq)),
                objectives=(), owns_inside=(), reads_outside=())),)))
            # every requirement held by the optimiser
            + Combine(SYSTEM, (SYSTEM, held, G1_NODE, G2_NODE, *BOUND_NODES.values()),
                      cx.RelationalCondition))


# ---------------------------------------------------------------- CO1, CO2 (Eqs. 9-10, Fig. 9)
#
# Collaborative optimisation, the objective taken whole. Nested: the system subproblem owns
# `z`, its copy `x̂1` of the local design variable and the targets `ŷ`, and minimises `f` at
# them. Each subproblem owns its own copy of `z` (and `x1`) and minimises its inconsistency
# `J_i` with the system subproblem under its own constraint. CO2's system subproblem holds each
# `J*_i = 0`; CO1's each consistency on its own, more equalities than unknowns, as Martins &
# Lambe warn.

def co(base, version: int = 2):
    J1, J2 = Inconsistency()(place('D1')), Inconsistency()(place('D2'))
    consistency = CONSISTENCY['D1'] + ((x1, x1_hat),) + CONSISTENCY['D2']
    system_holds = (
        (cx.Relation(J1, None, cx.Eq), cx.Relation(J2, None, cx.Eq)) if version == 2 else
        tuple(cx.Relation(a, b, cx.Eq) for a, b in consistency))
    held = flat.node('consistency')
    return (base
            # 1. the system subproblem's targets read in place of the couplings, its copy of
            #    x1, each subproblem's own copy of z; the declared optimiser determines nothing
            + Cut(y2, (D1_NODE, F2_NODE), cx.Copy())
            + Cut(y1, (D2_NODE, F1_NODE), cx.Copy())
            + Cut(x1, (F1_NODE,), cx.Copy())
            + Cut(z1, (D1_NODE,), Copy1()) + Cut(z2, (D1_NODE,), Copy1())
            + Cut(z1, (D2_NODE,), Copy2()) + Cut(z2, (D2_NODE,), Copy2())
            + Undetermine(SYSTEM)
            # 2. what the system subproblem holds of the subproblems, each subproblem and its
            #    inconsistency; each requirement into the problem owning its variables
            + insert(held, condition(system_holds))
            + insert(SUBPROBLEM_1, condition(objectives=(cx.Objective(J1, cx.Min),)))
            + insert(flat.node('inconsistency_1'), function_node(
                squared_distance, (z1_copy1, z1, z2_copy1, z2, x1_hat, x1, y1_hat, y1), (J1,)))
            + insert(SUBPROBLEM_2, condition(objectives=(cx.Objective(J2, cx.Min),)))
            + insert(flat.node('inconsistency_2'), function_node(
                squared_distance, (z1_copy2, z1, z2_copy2, z2, y2_hat, y2), (J2,)))
            + combine(SYSTEM, held, BOUND_NODES['z1'], BOUND_NODES['z2'])
            + combine(SUBPROBLEM_1, G1_NODE, BOUND_NODES['x1'])
            + combine(SUBPROBLEM_2, G2_NODE)
            # 3. the subproblems, then the system subproblem solving them in every evaluation
            + Determine(SUBPROBLEM_1, (z1_copy1, z2_copy1, x1))
            + Determine(SUBPROBLEM_2, (z1_copy2, z2_copy2))
            + Determine(SYSTEM, (z1, z2, x1_hat, y1_hat, y2_hat))
            + Nest(SUBPROBLEM_1, SYSTEM) + Nest(SUBPROBLEM_2, SYSTEM))


# ---------------------------------------------------------------- IPD, EPD (Eqs. 20-21, Fig. 14)
#
# Penalty decomposition, the objective as terms: Martins & Lambe require no `f0` and no `c0`,
# and Sellar's objective is `f1 + f2`. Nested: the system subproblem owns `z` and the targets
# `ŷ` and minimises the subproblems' penalties `Σ Φ*_i`, held only to the bounds of `z`, its
# own variables. Each subproblem owns its own copy of `z` (and `x1`) and minimises
# `f_i + Φ_i` under its own constraint, all in one node ("Discipline i and Penalty
# Functions"). IPD's `Φ_i` is a quadratic penalty; EPD's an L1 one over elastic parts, each
# consistency the difference `plus − minus` of two non-negative unknowns.

def _penalty_decomposition(base, exact: bool):
    terms = {'D1': (f1, (z1_copy1, z2_copy1, x1)), 'D2': (f2, (z1_copy2, z2_copy2))}
    subproblems, penalties, determines = [], [], []
    for d, (term, own) in terms.items():
        i = d[-1]
        sub = SUBPROBLEM_1 if i == '1' else SUBPROBLEM_2
        phi, objective = Penalty()(place(d)), SubproblemObjective()(place(f'subproblem_{i}'))
        pairs = CONSISTENCY[d]
        if not exact:
            penalties.append(insert(flat.node(f'penalty_{i}'), function_node(
                penalty_and_objective, (weight, term, *(p for pair in pairs for p in pair)),
                (phi, objective))))
            extra, holds = (), ()
        else:
            parts = [(ElasticPlus()(place(f'{d}_{k}')), ElasticMinus()(place(f'{d}_{k}')))
                     for k in range(len(pairs))]
            gaps  = tuple(ElasticResidual()(place(f'{d}_{k}')) for k in range(len(pairs)))
            penalties.append(insert(flat.node(f'penalty_{i}'), function_node(
                l1_penalty_with_barrier,
                (weight, barrier, term,
                 *(v for (a, b), (p, m) in zip(pairs, parts) for v in (a, b, p, m))),
                (phi, objective, *gaps))))
            extra = tuple(v for pm in parts for v in pm)
            holds = (*(cx.Relation(g, None, cx.Eq) for g in gaps),
                     *(cx.Relation(None, v, cx.Le) for v in extra))
        subproblems.append(insert(sub, condition(holds, (cx.Objective(objective, cx.Min),))))
        determines.append(Determine(sub, (*own, *extra)))
    system_objective = SubproblemObjective()(place('opt'))
    return (base
            # 1. the system subproblem's targets read in place of the couplings, each
            #    subproblem's own copy of z; the declared optimiser determines nothing and
            #    minimises Σ Φ*_i, so the declared sum of the objective terms goes
            + Cut(y2, (D1_NODE,), cx.Copy())
            + Cut(y1, (D2_NODE,), cx.Copy())
            + Cut(z1, (D1_NODE,), Copy1()) + Cut(z2, (D1_NODE,), Copy1())
            + Cut(z1, (D2_NODE,), Copy2()) + Cut(z2, (D2_NODE, F2_NODE), Copy2())
            + Undetermine(SYSTEM)
            + Rename(SYSTEM, ((f, system_objective),))
            + Delete((F_NODE,))
            # 2. the system subproblem's sum, each subproblem and its penalty functions; each
            #    requirement into the problem owning its variables
            + insert(flat.node('penalty_sum'), function_node(
                add, (Penalty()(place('D1')), Penalty()(place('D2'))), (system_objective,)))
            + subproblems[0] + penalties[0] + subproblems[1] + penalties[1]
            + combine(SYSTEM, BOUND_NODES['z1'], BOUND_NODES['z2'])
            + combine(SUBPROBLEM_1, G1_NODE, BOUND_NODES['x1'])
            + combine(SUBPROBLEM_2, G2_NODE)
            # 3. the subproblems, then the system subproblem solving them in every evaluation
            + determines[0] + determines[1]
            + Determine(SYSTEM, (z1, z2, y1_hat, y2_hat))
            + Nest(SUBPROBLEM_1, SYSTEM) + Nest(SUBPROBLEM_2, SYSTEM))


def ipd(base): return _penalty_decomposition(base, exact=False)
def epd(base): return _penalty_decomposition(base, exact=True)


# ---------------------------------------------------------------- ATC (Eqs. 18-19, Fig. 13)
#
# Analytical target cascading with a quadratic penalty, the objective as terms (`f0 = 0`, no
# `c0`, so no `Φ0`). The system subproblem and the subproblems are solved in turn (Algorithm
# 7), so the top is a fixed point owning copies of what the subproblems read of the system
# subproblem and holding their consistency, the subproblems nested in it. The system
# subproblem owns `z` and the targets `ŷ` and minimises its penalties ("System and Penalty
# Functions"), reading the subproblems' copies and responses as fixed (Fig. 13: "nonlocal data
# are always treated as fixed values"). Each subproblem owns its own copy of `z` (and `x1`) and
# minimises `f_i + Φ_i` under its own constraint ("Discipline i and Penalty Functions"). The
# top's cut moves only the subproblems' reads of the system subproblem's variables onto its
# copies.

def atc(base):
    def subproblem_penalty(d, term):
        i = d[-1]
        pairs = (p for pair in CONSISTENCY[d] for p in pair)
        return (flat.node(f'penalty_{i}'), function_node(
            penalty_and_objective, (weight, term, *pairs),
            (Penalty()(place(d)), SubproblemObjective()(place(f'subproblem_{i}')))))
    every_pair = tuple(p for d in ('D1', 'D2') for pair in CONSISTENCY[d] for p in pair)
    objective_1, objective_2, system_objective = (
        SubproblemObjective()(place(at)) for at in ('subproblem_1', 'subproblem_2', 'opt'))
    penalties = (flat.node('penalty_1'), flat.node('penalty_2'))
    top = flat.node('top')
    # the top's copies of what the subproblems read of the system subproblem, each beside it
    held_fixed = tuple((cx.Copy()(v), v) for v in (z1, z2, y1_hat, y2_hat))
    return (base
            # 1. the system subproblem's targets read in place of the couplings, each
            #    subproblem's own copy of z; the declared optimiser determines nothing and
            #    minimises its penalties, so the declared sum of the objective terms goes
            + Cut(y2, (D1_NODE,), cx.Copy())
            + Cut(y1, (D2_NODE,), cx.Copy())
            + Cut(z1, (D1_NODE,), Copy1()) + Cut(z2, (D1_NODE,), Copy1())
            + Cut(z1, (D2_NODE,), Copy2()) + Cut(z2, (D2_NODE, F2_NODE), Copy2())
            + Undetermine(SYSTEM)
            + Rename(SYSTEM, ((f, system_objective),))
            + Delete((F_NODE,))
            # 2. the system subproblem's penalty functions, each subproblem and its penalty
            #    functions; each requirement into the problem owning its variables
            + insert(flat.node('penalty_system'), function_node(
                penalty, (weight, *every_pair), (system_objective,)))
            + insert(SUBPROBLEM_1, condition(objectives=(cx.Objective(objective_1, cx.Min),)))
            + insert(*subproblem_penalty('D1', f1))
            + insert(SUBPROBLEM_2, condition(objectives=(cx.Objective(objective_2, cx.Min),)))
            + insert(*subproblem_penalty('D2', f2))
            + combine(SYSTEM, BOUND_NODES['z1'], BOUND_NODES['z2'])
            + combine(SUBPROBLEM_1, G1_NODE, BOUND_NODES['x1'])
            + combine(SUBPROBLEM_2, G2_NODE)
            # 3. the problems solved in turn; then the top over them: the subproblems' reads
            #    of the system subproblem's variables cut, the fixed point owning the copies,
            #    and the problems nested in it
            + Determine(SYSTEM, (z1, z2, y1_hat, y2_hat))
            + Determine(SUBPROBLEM_1, (z1_copy1, z2_copy1, x1))
            + Determine(SUBPROBLEM_2, (z1_copy2, z2_copy2))
            + Cut(z1, penalties, cx.Copy())
            + Cut(z2, penalties, cx.Copy())
            + Cut(y1_hat, (D2_NODE, penalties[0]), cx.Copy())
            + Cut(y2_hat, (D1_NODE, penalties[1]), cx.Copy())
            + insert(top, condition(tuple(cx.Relation(c, v, cx.Eq) for c, v in held_fixed)))
            + Determine(top, tuple(c for c, _ in held_fixed))
            + Nest(SYSTEM, top) + Nest(SUBPROBLEM_1, top) + Nest(SUBPROBLEM_2, top))


# ---------------------------------------------------------------- QSD (Eqs. 23-24, Fig. 16)
#
# Quasi-separable decomposition, the objective as terms, `D1` its only discipline: `D2` has no
# local design variable, so it is the system level's, `f2` and `g2` its system terms, and `y2`
# is not copied. Nested: the system subproblem owns `z`, the target `ŷ1` and the budget `b1`,
# minimises `f2 + b1` and holds `g2 ≤ 0` and `0 ≤ s1`. The subproblem owns `x1` and its margin
# `s1`, minimises `−s1`, and holds `g1 + s1 ≤ 0`, `f1 − b1 + s1 ≤ 0`, `x1`'s bounds loosened
# alike (`s1 ≤ x1`, `x1 + s1 ≤ 10`), and `ŷ1 = y1`, its margins one node ("Discipline i
# Functions"). Every local constraint is held through the margin (Eq. 24: `c_ik − s_i ≥ 0`),
# `x1`'s bounds included, so the subproblem is feasible at every system point although
# `ŷ1 = y1` fixes `x1`; their declared requirements go. A problem reads its objective and may
# not read what it owns, so `−s1` is that node's output, as Martins & Lambe write it ("minimize
# −s_i").

def qsd(base):
    b1, s1 = Budget()(place('D1')), Margin()(place('D1'))
    g1_loose, f1_loose, x1_loose = Loosened()(g1), Loosened()(f1), Loosened()(x1)
    upper_x1 = place(cx.upper('x1'))
    objective_1, system_objective = (SubproblemObjective()(place(at))
                                     for at in ('subproblem_1', 'opt'))
    margin = flat.node('margin')
    return (base
            # 1. the system subproblem's target read in place of the coupling; the declared
            #    optimiser determines nothing and minimises f2 + b1, so the declared sum of the
            #    objective terms goes, and the requirements the margins replace
            + Cut(y1, (D2_NODE,), cx.Copy())
            + Undetermine(SYSTEM)
            + Rename(SYSTEM, ((f, system_objective),))
            + Delete((F_NODE, G1_NODE, BOUND_NODES['x1']))
            # 2. the margin the system subproblem holds and its sum, the subproblem and its
            #    margins; each requirement into the problem owning its variables
            + insert(margin, condition((cx.Relation(None, s1, cx.Le),)))
            + insert(flat.node('budget_sum'), function_node(add, (f2, b1), (system_objective,)))
            + insert(SUBPROBLEM_1, condition(
                (cx.Relation(g1_loose, None, cx.Le), cx.Relation(f1_loose, None, cx.Le),
                 cx.Relation(s1, x1, cx.Le), cx.Relation(x1_loose, upper_x1, cx.Le),
                 cx.Relation(y1_hat, y1, cx.Eq)),
                (cx.Objective(objective_1, cx.Min),), (y1_hat, upper_x1)))
            + insert(flat.node('margins_1'), function_node(
                qsd_margins, (g1, f1, b1, x1, s1), (g1_loose, f1_loose, x1_loose, objective_1)))
            + combine(SYSTEM, margin, G2_NODE, BOUND_NODES['z1'], BOUND_NODES['z2'])
            # 3. the subproblem, then the system subproblem solving it in every evaluation
            + Determine(SUBPROBLEM_1, (x1, s1))
            + Determine(SYSTEM, (z1, z2, y1_hat, b1))
            + Nest(SUBPROBLEM_1, SYSTEM))


# ---------------------------------------------------------------- ASO (Eqs. 25-26, Fig. 17)
#
# Asymmetric subspace optimisation: one discipline optimised inside the MDA, the other only
# analysed there. Sellar's `D2` has no local design variable, so `D1` is optimised. The system
# optimiser owns `z`, minimises `f`, and holds `g2`, the constraint of the discipline not
# optimised. The MDA owns the copy `ŷ1`, read by `D2`, and holds `ŷ1 = y1`. The subproblem,
# nested in the MDA, owns `x1`, minimises the system optimiser's own `f` with `ŷ1` fixed, and
# holds `g1`. Each MDA iteration runs `D2`, the analysed discipline, then the subproblem, each
# of whose iterations runs `D1` alone, as Martins & Lambe's Algorithm 11 runs them.

def aso(base):
    mda = flat.node('mda')
    return (base
            # 1. D2 reads the copy ŷ1 in place of the coupling; the declared optimiser
            #    determines nothing
            + Cut(y1, (D2_NODE,), cx.Copy())
            + Undetermine(SYSTEM)
            # 2. the MDA's ŷ1 = y1, the subproblem minimising f; each requirement into the
            #    problem owning its variables
            + insert(mda, condition((cx.Relation(y1_hat, y1, cx.Eq),)))
            + insert(SUBPROBLEM_1, condition(objectives=(cx.Objective(f, cx.Min),)))
            + combine(SYSTEM, G2_NODE, BOUND_NODES['z1'], BOUND_NODES['z2'])
            + combine(SUBPROBLEM_1, G1_NODE, BOUND_NODES['x1'])
            # 3. the subproblem over x1, the MDA over ŷ1 around it, the system optimiser over
            #    z around the MDA
            + Determine(SUBPROBLEM_1, (x1,))
            + Determine(mda, (y1_hat,)) + Nest(SUBPROBLEM_1, mda)
            + Determine(SYSTEM, (z1, z2)) + Nest(mda, SYSTEM))


# ---------------------------------------------------------------- BLISS-2000 (Eqs. 16-17, Fig. 12)
#
# NOT THE SAME METHOD. Published BLISS-2000 exchanges surrogate models of the subproblem
# optima, and Martins & Lambe call it equivalent only "provided the weighting coefficients
# accurately capture the influence of the coupling variables on the objective". This plan
# solves the subproblems exactly instead of through surrogates, and Sellar's `D2`, with no
# local design variable, has no subproblem: it is evaluated at the system subproblem, which
# holds its `g2`. Nested: the system subproblem owns `z`, the targets `ŷ` and the weight `w1`,
# minimises `f` at the subproblem's optimum, and holds `ŷ_i = y_i`. The subproblem owns `x1`,
# minimises `w1 · y1`, and holds `g1`.

def bliss2000(base):
    w1, objective_1 = Weight()(place('D1')), SubproblemObjective()(place('subproblem_1'))
    held = flat.node('consistency')
    return (base
            # 1. the system subproblem's targets read in place of the couplings; the declared
            #    optimiser determines nothing
            + Cut(y2, (D1_NODE,), cx.Copy())
            + Cut(y1, (D2_NODE,), cx.Copy())
            + Undetermine(SYSTEM)
            # 2. the consistency the system subproblem holds, the subproblem and its weighted
            #    response; each requirement into the problem owning its variables
            + insert(held, condition((cx.Relation(y1_hat, y1, cx.Eq),
                                      cx.Relation(y2_hat, y2, cx.Eq))))
            + insert(SUBPROBLEM_1, condition(objectives=(cx.Objective(objective_1, cx.Min),)))
            + insert(flat.node('weighted_1'), function_node(multiply, (w1, y1), (objective_1,)))
            + combine(SYSTEM, held, G2_NODE, BOUND_NODES['z1'], BOUND_NODES['z2'])
            + combine(SUBPROBLEM_1, G1_NODE, BOUND_NODES['x1'])
            # 3. the subproblem, then the system subproblem, owning the weight too, solving it
            #    in every evaluation
            + Determine(SUBPROBLEM_1, (x1,))
            + Determine(SYSTEM, (z1, z2, y1_hat, y2_hat, w1))
            + Nest(SUBPROBLEM_1, SYSTEM))


# ---------------------------------------------------------------- how each plan runs
#
# Whether the system subproblem takes finite differences (it reads a nested constrained
# solve), the starts of the unknowns `START` does not hold, the inputs a plan adds, the outer
# loop of a penalty method, and how close the answer must come.

COPIES_OF_Z = {z1_copy1: 5.0, z2_copy1: 2.0, z1_copy2: 5.0, z2_copy2: 2.0}


@dataclasses.dataclass(frozen=True)
class Run:
    differences : bool = False
    '''Whether the system subproblem's SLSQP takes finite differences.'''
    starts : dict = dataclasses.field(default_factory=dict)
    inputs : dict = dataclasses.field(default_factory=dict)
    tol : float = 1e-4
    strict : bool = True
    '''Whether a nested subproblem that does not converge stops the run.'''
    system_strict : bool = True
    '''Whether a system subproblem that does not meet its own convergence test stops the
    run.'''
    system_tol : float = 1e-7
    '''The system subproblem's SLSQP tolerance.'''
    stages : tuple[dict, ...] = ()
    '''A penalty method's outer loop: the inputs of each stage, each run warm-started from the
    last.'''
    names : dict = dataclasses.field(default_factory=dict)
    '''How the drawing writes a name this plan means differently.'''
    note : str = ''
    '''What the overview says beside its result.'''

    @property
    def solvers(self) -> dict:
        return dict(differences=self.differences, strict=self.strict,
                    system_strict=self.system_strict, system_tol=self.system_tol)

    def answer(self, model):
        '''`model` run as this says: its answer at each of the optimum's names.'''
        return run_staged(model, starts=self.starts, inputs=self.inputs, stages=self.stages,
                          **self.solvers)

    def xdsm(self, model):
        '''`model`'s XDSM, with the solvers this runs it with.'''
        return xdsm(model, self.names, **self.solvers)


# WORKAROUND(FR-12): the outer loops as data, run outside the graph.
WEIGHTS = tuple({weight: w} for w in (1.0, 10.0, 100.0, 1000.0, 10000.0))
'''Martins & Lambe's weight update (Algorithms 7, 8): the weight raised tenfold per stage.'''

BARRIERS = tuple({barrier: mu} for mu in (1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-6))
'''EPD's barrier on its elastic variables, lowered tenfold per stage, its weight fixed.'''

ELASTIC_STARTS = {v: 1.0 for d in ('D1', 'D2') for k in range(3)
                  for v in (ElasticPlus()(place(f'{d}_{k}')), ElasticMinus()(place(f'{d}_{k}')))}


RUNS = {
    'SAND':       Run(),
    'IDF':        Run(),
    'AAO':        Run(),
    'MDF':        Run(),
    'CO1':        Run(True, COPIES_OF_Z, tol=1e-3),
    'CO2':        Run(True, COPIES_OF_Z, tol=2e-3),
    'IPD':        Run(True, COPIES_OF_Z, tol=5e-3, stages=WEIGHTS),
    'EPD':        Run(True, {**COPIES_OF_Z, **ELASTIC_STARTS}, {weight: 10.0}, tol=1e-3,
                      stages=BARRIERS),
    'ATC':        Run(False, {**COPIES_OF_Z, cx.Copy()(y1_hat): 1.0, cx.Copy()(y2_hat): 1.0},
                      tol=1e-3, stages=WEIGHTS),
    'QSD':        Run(True, {Budget()(place('D1')): 4.0, Margin()(place('D1')): 0.0},
                      tol=1e-3, system_strict=False,
                      names={'^subproblemobjective.opt': r'f_2 + b_1',
                             '^subproblemobjective.subproblem_1': r'-s_1'},
                      note='system subproblem stopped at its iteration limit'),
    'ASO':        Run(True, tol=1e-3),
    'BLISS-2000': Run(True, {Weight()(place('D1')): 1.0}, tol=1e-3,
                      names={'^subproblemobjective.subproblem_1': r'w_1 y_1'},
                      note='subproblems solved exactly, without surrogates'),
}
'''Each plan by its name, and how it runs.'''
