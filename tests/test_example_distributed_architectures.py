'''The Sellar plans (`docs/source/examples/distributed_architectures.py`) against the
current API: every recipe in `RUNS` built, driven by the example's solvers, given its values
and drawn, as the notebook does it, without solving anything. Whether each converges is the
notebook's.'''

import sys
from pathlib import Path

import jax
import pytest

jax.config.update('jax_enable_x64', True)
pytest.importorskip('scipy')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'docs' / 'source' / 'examples'))

import cottax.assemblies.jax as cx                                    # noqa: E402
from cottax.assemblies.jax import flat                                # noqa: E402
from sellar_helpers import START, declared_xdsm, sellar, with_solvers  # noqa: E402
from distributed_architectures import (                               # noqa: E402
    RUNS, aao, aso, atc, bliss2000, co, epd, idf, ipd, mdf, qsd, sand,
)

RECIPES = {
    'SAND': sand, 'IDF': idf, 'AAO': aao, 'MDF': mdf,
    'CO1': lambda base: co(base, version=1), 'CO2': co,
    'IPD': ipd, 'EPD': epd, 'ATC': atc, 'QSD': qsd, 'ASO': aso, 'BLISS-2000': bliss2000,
}
'''Each plan of `RUNS` and the recipe the notebook builds it with.'''


def test_every_run_has_a_recipe():
    assert RECIPES.keys() == RUNS.keys()


def test_the_declared_problem_draws():
    assert declared_xdsm(sellar()).startswith('<svg')


@pytest.mark.parametrize('name', RUNS)
def test_a_recipe_builds_drives_takes_its_values_and_draws(name):
    how = RUNS[name]
    model = RECIPES[name](sellar())
    inputs = {**how.inputs, **(how.stages[0] if how.stages else {})}
    values = {key: given for key, given in ((cx.Start, how.starts), (cx.Inputs, inputs))
              if given}
    flat.valued(with_solvers(model, **how.solvers), values, read_from={cx.Start: dict(START)})
    assert how.xdsm(model).startswith('<svg')
