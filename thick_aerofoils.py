# import modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# import high speed solver
import utils

# Inputs class
class Inputs:

    # NACA aerofoil text files
    filenames = ["naca0009.txt", "naca0012.txt"]
    filename = "naca0009.txt"

    # default parameters
    N = 10000
    sharpness = 100
    thickness_fraction = 0.5

# Aerofoils class
class Aerofoils:
    """Contains helper functions for imposing thickness restrictions on a NACA aerofoil."""

    def __init__(self):

        pass

    def load_aerofoil(self, filename):
        """Loads an aerofoil x- and y-coordinates from a .txt file."""
        # load data and separate into x- and y-components of upper surface
        data = np.loadtxt(filename, skiprows = 1)
        z = np.array([x for x in data if x[1] >= 0])
        x = np.array(z[:-1, 0])[::-1]
        y = np.array(z[:-1, 1])[::-1]

        # fit quadratic spline and return as finely-spaced datapoints
        spline = make_interp_spline(x, y, k = 2)
        x_fine = np.linspace(x.min(), x.max(), Inputs.N)
        y_fine = spline(x_fine)
        aerofoil_data = np.array([x_fine, y_fine])
        return aerofoil_data
    
    def thick_aerofoil(
            self, thickness_fraction = Inputs.thickness_fraction, filename = Inputs.filename,
            sharpness = Inputs.sharpness
        ):
        """Imposes thickness restrictions on a given aerofoil."""
        # load aerofoil data
        data = self.load_aerofoil(filename)

        # find y-value of min. thickness
        t_min = np.max(data[1]) * thickness_fraction

        # find x-coordinate (scalar) and y-coordinates (vector) of all possible circles
        c_x = 1 - t_min
        c_y = ((c_x - data[0])**2 + data[1]**2 - t_min**2) / (2 * (data[1] - t_min))

        # get gradient of circle radius arms
        m = (c_y - data[1]) / (c_x - data[0])

        # get gradient of aerofoil
        dy_dx = np.gradient(data[1], data[0])

        # get tangency residuals
        dtan = dy_dx + 1 / m

        # get index of closest to zero tangency residual
        tangent_index = np.argmin(np.abs(dtan))

        # get index of rear fillet start
        fillet_index = np.argmin(np.abs(data[0] + t_min - 1))

        # replace data between fillets with that of circle
        r = c_y[tangent_index] - t_min
        data[1][tangent_index:fillet_index] = c_y[tangent_index] - np.sqrt(r**2 - (data[0][tangent_index:fillet_index] - c_x)**2)

        # get trailing edge circle datapoints
        data[1][fillet_index:] = np.sqrt(np.maximum(t_min**2 - (data[0][fillet_index:] + t_min - 1)**2, 0))

        # store circles for visualisation
        self.circle_1 = plt.Circle((c_x, c_y[tangent_index]), r, fill=False)
        self.circle_2 = plt.Circle((1 - t_min, 0), t_min, fill=False)

        return data
    
    def thick_aerofoil_backup(
            self, thickness_fraction = Inputs.thickness_fraction, filename = Inputs.filename,
            sharpness = Inputs.sharpness
        ):
        """Imposes thickness restrictions on a given aerofoil."""
        # load aerofoil data
        data = self.load_aerofoil(filename)
        
        # find index of max. thickness
        index = np.argmax(data[1])

        # find y-value of min. thickness
        t_min = np.max(data[1]) * thickness_fraction

        # clip y-data from point of max. thickness to trailing edge to impose minimum thickness
        data[1][index:] = (
            utils.soft_clip(data[1][index:], a_min = t_min, a_max = None, sharpness = sharpness)
        )

        # over final chord region, incrementally bleed original
        diff = data[1] - (1 - data[0])
        sign_changes = np.where(np.diff(np.sign(diff)))[0]
        fillet_index = sign_changes[-1]

        # get y-coordinate
        y_fillet = data[1][fillet_index]
        data[1][fillet_index:] = np.sqrt(
            np.maximum(y_fillet**2 - (data[0][fillet_index] - data[0][fillet_index:])**2, 0)
        )

        return data

# main function
def main():

    # create plot
    fig, ax = plt.subplots(figsize = (14, 7))
    ax.grid()
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.2, 0.3)
    aerofoils = Aerofoils()
    for filename in Inputs.filenames:

        aerofoil_data = aerofoils.load_aerofoil(filename)
        ax.plot(aerofoil_data[0], aerofoil_data[1], label = f"{filename}")

    # generate thickened aerofoil profile
    aerofoil_data = aerofoils.thick_aerofoil()

    # plot circles used as fillets
    ax.add_patch(aerofoils.circle_1)
    ax.add_patch(aerofoils.circle_2)

    ax.plot(aerofoil_data[0], aerofoil_data[1], label = f"Thickened")

    # add legend
    ax.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
    plt.tight_layout()

if __name__ == "__main__":

    main()
    plt.show()
