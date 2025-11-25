# import modules

import numpy as np
import matplotlib.pyplot as plt
import airfoils as af
from pprint import pprint
from scipy.interpolate import make_interp_spline

import xml_exporter as xml

import sys

# create Defaults class
class Defaults:
    """Container for default values."""
    name = "4812"
    length = 1
    label = False
    verbose = False
    fill = False
    N = 40

def main():
    """Creates an instance of the aerofoil class and passes it to xml_exporter."""
    aerofoil = af.Airfoil.NACA4(Defaults.name)

    if Defaults.verbose:
            
        # print all aerofoil methods and attributes
        print([method_name for method_name in dir(aerofoil)
                    if callable(getattr(aerofoil, method_name))])
        pprint(vars(aerofoil))
        aerofoil.plot()

    # create array of x-values
    theta = np.linspace(0, np.pi, Defaults.N)
    x = 0.5 * (1 - np.cos(theta))

    # fit splines and store y-values of upper and lower surfaces
    spline = make_interp_spline(aerofoil._x_upper, aerofoil._y_upper, k = 1)
    yy_upper = spline(x)
    spline = make_interp_spline(aerofoil._x_lower, aerofoil._y_lower, k = 1)
    yy_lower = spline(x)
    yy = np.concatenate((
        yy_upper[::-1],
        yy_lower
    ))
    xx = np.concatenate((x[::-1], x))

    # store data as x-y pairs
    data = np.transpose([xx, yy])

    # scale data
    data *= Defaults.length

    # create plot showing finished result
    fig, ax = plt.subplots()
    ax.plot(
        [xy[0] for xy in data],
        [xy[1] for xy in data],
        linestyle = '', marker = '.', markersize = 2
    )
    ax.grid()
    ax.axis('equal')
    plt.show()
    
    # use xml_exporter to produce XML file
    xml.xml_exporter(data, Defaults.label)

# on script execution
if __name__ == "__main__":

    # try to read input arguments
    try:

        if len(sys.argv) > 1:

            # read name
            Defaults.name = str(sys.argv[1])

        if len(sys.argv) > 2:

            # read length and convert to float
            Defaults.length = float(sys.argv[2])

        if len(sys.argv) > 3:

            # read label
            Defaults.label = str(sys.argv[3])

        if 'v' in sys.argv:

            # set verbose to True
            Defaults.verbose = True

        if 'fill' in sys.argv:

            # set fill to True
            Defaults.fill = True

    # report error
    except Exception as error:

        print(
            f"{xml.Colours.RED}Please use the following syntax:{xml.Colours.END}\n"
            f"python aerofoils.py [{xml.Colours.CYAN}4xxx{xml.Colours.END}] "
            f"[{xml.Colours.CYAN}length{xml.Colours.END}] "
            f"[{xml.Colours.CYAN}label{xml.Colours.END}] "
            f"[{xml.Colours.CYAN}v{xml.Colours.END}]\n"
            f"{error}"
        )

    # run main() on script execution
    main()