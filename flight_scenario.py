# import modules

import numpy as np
import utils
import ambiance

# create Flight_scenario class

class Flight_scenario:
    """
    Represents a flight scenario with dimensional parameters.

    Stores all variables relevant to a given aeroplane operation point with specified altitude,
    flight velocity, and engine diameter.

    Parameters
    ----------
    altitude : float
        Flight altitude in metres.
    velocity : float
        Flight velocity in m/s.
    diameter : float
        Engine diameter at inlet to the rotor of the first stage in metres.
    hub_tip_ratio : float
        Hub-to-tip ratio of the engine at inlet to the rotor of the first stage.
    thrust : float
        Target engine thrust in Newtons.
    """
    def __init__(self, altitude, velocity, diameter, hub_tip_ratio, thrust):
        """Create instance of the Flight_scenario class."""
        # save input parameters
        self.altitude = altitude
        self.velocity = velocity
        self.diameter = diameter
        self.hub_tip_ratio = hub_tip_ratio
        self.thrust = thrust

        # determine atmospheric properties at specified altitude
        self.atmosphere = ambiance.Atmosphere(self.altitude)
        self.T = self.atmosphere.temperature
        self.p = self.atmosphere.pressure
        self.rho = self.atmosphere.density
        self.a = self.atmosphere.speed_of_sound

        # calculate derived non-dimensional quantities
        self.M = self.velocity / self.a
        self.A = (np.pi / 4) * self.diameter**2
        self.p_0 = self.p / utils.stagnation_pressure_ratio(self.M)
        self.C_th = self.thrust / (self.A * self.p_0)

    def __str__(self):
        """Prints a string representation of the flight scenario."""
        string = f"\n"
        for name, value in self.__dict__.items():

            if isinstance(value, (int, float)):

                if ("alpha" in name or "angle" in name or "beta" in name) and not value == 0:

                    string += f"{name}: {utils.Colours.GREEN}{utils.rad_to_deg(value):.4g} °"
                    string += f"{utils.Colours.END}\n"

                else:

                    string += f"{name}: {utils.Colours.GREEN}{value:.4g}{utils.Colours.END}\n"

        string += "\n"

        return string