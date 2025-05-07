import pytest

from project_name.pyfuncs import my_example_multiplier

def test_my_multipler():
    """
    Test basic use case of my_multiplier
    """

    assert my_example_multiplier(1, 2, 3) == 6

def test_my_multiplier_raises():
    """
    Confirm that we do not have input data sanitation in
    my_multiplier
    """

    with pytest.raises(TypeError):
        my_example_multiplier(None, 1, 2)