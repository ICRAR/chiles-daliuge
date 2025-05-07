"""
project_name pyfuncs module.

This is the starting module of project_name containing DALiuGE PyFuncs.
Here you can start putting the functions you are building for a DALiuGE Palette.

Typically a PyFunc project will contain multiple PyFuncs and will
then result in a single EAGLE palette.

Be creative! do whatever you need to do!
"""

def my_example_multiplier(x: int = 1, y: int = 2, n: int = 3):
    """
    My custom multiplier function.

    Parameters
    ----------
    x: The first multiplier
    y: The second multiplier
    n: The number of times we multiple x*y!

    Returns
    -------
    The multiple of  n * (x * y)
    """

    return x*y*n