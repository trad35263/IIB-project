# import modules
import numpy as np
import matplotlib.pyplot as plt

# create Coefficients class
class Coefficients:
    """Stores polynomial coefficients in descending powers."""

    def __init__(self, coefficients):
        """Creates an instance of the coefficients class."""
        # store input variables
        self.coefficients = np.array(coefficients, copy=True)

    def calculate(self, x, y):
        """Fit a polynomial to (x, y) data and stores the coefficient in descending powers."""
        # calculate order from initiialised coefficients array and fit polynomial
        order = self.coefficients.size - 1
        self.coefficients[:] = np.polyfit(x, y, order)
