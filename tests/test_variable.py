import pytest
import numpy as np

from desdeo_problem.problem import IntegerVariable, VariableError


@pytest.mark.variable
class TestIntegerVariable:
    def test_init_float_lower_bound(self):
        """Test that float value for lower bound that is not -inf raises an exception.
        """
        lower_bound = 10.1

        with pytest.raises(VariableError) as err:
            var = IntegerVariable("name", 15, lower_bound=lower_bound)

        assert "must be of type int" in str(err.value)

    def test_init_float_upper_bound(self):
        """Test that float value for upper bound that is not inf raises and exception.
        """
        upper_bound = 20.12

        with pytest.raises(VariableError) as err:
            var = IntegerVariable("name", 15, upper_bound=upper_bound)

        assert "must be of type int" in str(err.value)

    def test_init_bad_initial_value(self):
        """Test that an initial value outside the given lower and upper bounds raises an exception.
        """
        lower_bound = 5
        upper_bound = 20
        initial_value = 21

        with pytest.raises(VariableError) as err:
            var = IntegerVariable("name", initial_value, lower_bound=lower_bound, upper_bound=upper_bound)

        assert "must be between" in str(err.value)

    def test_init_float_initial_value(self):
        """Test that an initial value which is not an int raises an exception.
        """
        initial_value = 5.2

        with pytest.raises(VariableError) as err:
            var = IntegerVariable("name", initial_value)

        assert "must be of type 'int'" in str(err.value)

    def test_init_bounds_wrong(self):
        """Test that when a lower bound that is greater than the upper bound raises an exception.
        """
        lower_bound = 69
        upper_bound = 42
        initial_value = 50

        with pytest.raises(VariableError) as err:
            var = IntegerVariable("name", initial_value, lower_bound=lower_bound, upper_bound=upper_bound)

        assert "must be less than" in str(err.value)

    def test_get_bounds(self):
        """Test that bounds are returned properly
        """
        lower_bound = 42
        upper_bound = 101
        initial_value = 82

        var = IntegerVariable("name", initial_value, lower_bound, upper_bound)

        assert var.get_bounds() == (lower_bound, upper_bound)

    def test_get_default(self):
        """Test that default bounds are returned correctly.
        """
        initial_value = 82

        var = IntegerVariable("name", initial_value)

        assert var.get_bounds() == (float("-inf"), float("inf"))
