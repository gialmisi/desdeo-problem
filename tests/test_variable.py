from desdeo_problem.Variable import (Variable, VariableBuilderError,
                                     VariableError, variable_builder)
import pytest
from pytest import approx

import numpy as np

name = "test_name"
initial_value = 1.5
lower_bound = 0.5
upper_bound = 2.55


@pytest.fixture
def simple_variable():
    """Create a simple variable

    """
    simple_variable = Variable(name,
                               initial_value,
                               lower_bound,
                               upper_bound)
    return simple_variable


def test_simple_variables(simple_variable):
    """Test an initialized variable for proper values.

    """
    assert simple_variable.name == name
    assert simple_variable.initial_value == approx(initial_value)
    assert simple_variable.current_value == approx(initial_value)


def test_get_bounds(simple_variable):
    """Test get bounds

    """
    get_low, get_up = simple_variable.get_bounds()
    assert get_low == approx(lower_bound)
    assert get_up == approx(upper_bound)


def test_bad_bounds():
    """Test creating a variable with ill defined bounds

    """
    with pytest.raises(VariableError):
        Variable("bad_variable", 0, 10.0, 1.0)


def test_bad_initial_value():
    """Test creating a variable with an ill defined initial value.

    """
    with pytest.raises(VariableError):
        Variable("bad_variable", -1, 5.0, 10.0)


def test_variable_builder():
    """Test the variable builder

    """
    names = ["x1", "x2", "x3"]
    inits = [1.1, 2.2, 3.3]
    lows = [0.1, 0.2, 0.3]
    ups = [10.1, 10.2, 10.3]
    variables = variable_builder(names, inits, lows, ups)

    for (i, var) in enumerate(variables):
        assert var.name == names[i]
        assert var.initial_value == approx(inits[i])

        low, up = var.get_bounds()
        assert low == approx(lows[i])
        assert up == approx(ups[i])


def test_variable_builder_edge_cases():
    """Test the creation of variables with the variable builder in edge cases

    """
    variable_one = variable_builder(["test"], [1], [0.1], [10.0])
    assert len(variable_one) == 1

    variable_nil = variable_builder([], [], [], [])
    assert len(variable_nil) == 0


def test_variable_builder_defaults():
    """Test the default values of the variable_builder

    """
    names = ["x1", "x2", "x3"]
    inits = [1.1, 2.2, 3.3]
    ups = [10.1, 10.2, 10.3]

    # no low bounds given
    variables_low_infs = variable_builder(names, inits, None, ups)
    for (i, var) in enumerate(variables_low_infs):
        low, _ = var.get_bounds()
        assert np.isclose(low, -np.inf)

    lows = [0.1, 0.2, 0.3]

    # no upper bounds given
    variables_up_infs = variable_builder(names, inits, lows, None)
    for (i, var) in enumerate(variables_up_infs):
        _, up = var.get_bounds()
        assert np.isclose(up, np.inf)


def test_variable_builder_errors():
    """Test the cases that should raise an error

    """
    # wrong number of initial values
    with pytest.raises(VariableBuilderError):
        variable_builder(["test"], [1.0, 2.0])

    # wrong number of lower bounds
    with pytest.raises(VariableBuilderError):
        variable_builder(["test"], [1.0], [0.1, 0.2])

    # wrong number of upper bounds
    with pytest.raises(VariableBuilderError):
        variable_builder(["test"], [1.0], None, [10.0, 20.0])
