# import modules
import pytest
import sys
from pathlib import Path
import utils
from time import perf_counter as timer
import numpy as np

# adjust sys.path to import your modules
sys.path.insert(0, str(Path(__file__).parent))

from engine import Engine
from flight_scenario import Flight_scenario
import utils

# Assert class
class Assert:
    """Stores several functions relating to engine evaluation."""
    def no_nans(engine):
        """Checks for nans in an engine."""
        # loop over each blade row
        for blade_row in engine.blade_rows:

            # loop for inlet and exit
            for annulus in [blade_row.inlet, blade_row.exit]:

                # assert no nans in any primary variables
                assert np.any(np.isnan(annulus.M)) == False
                assert np.any(np.isnan(annulus.alpha)) == False
                assert np.any(np.isnan(annulus.T_0)) == False
                assert np.any(np.isnan(annulus.p_0)) == False

# TestEngine class
class TestEngine:
    """Test suite for engine creation and basic configuration."""
    def create_engine(
            self,
            altitude = utils.Defaults.altitude,
            flight_speed = utils.Defaults.flight_speed,
            diameter = utils.Defaults.diameter,
            hub_tip_ratio = utils.Defaults.hub_tip_ratio,
            thrust = utils.Defaults.thrust,
            no_of_stages = utils.Defaults.no_of_stages,
            phi = utils.Defaults.phi,
            psi = utils.Defaults.psi,
            vortex_exponent = utils.Defaults.vortex_exponent,
            Y_p = utils.Defaults.Y_p
        ):
        """Creates an engine with custom parameters."""
        # start timer
        t1 = timer()
        
        # create flight scenario
        flight_scenario = Flight_scenario(
            "", altitude, flight_speed, diameter, hub_tip_ratio, thrust
        )
        
        # create engine
        engine = Engine(
            flight_scenario, no_of_stages, phi, psi, vortex_exponent, Y_p
        )

        print(f"flight_scenario: {flight_scenario}")
        print(f"engine: {engine}")

        # end timer
        t2 = timer()
        print(f"Test engine created in {utils.Colours.GREEN}{t2 - t1:.4g}{utils.Colours.END} s.")
        
        return engine

    def test_engine_creation(self):
        """Test that an engine can be created with default parameters."""
        # create default engine
        engine = self.create_engine()

        # test criteria
        assert len(engine.stages) > 0
        assert len(engine.blade_rows) > 1
    
        # check for nans
        assert np.any(np.isnan(engine.blade_rows[-1].exit.M)) == False

    def test_engine_evaluation(self):
        """Test that engine performance is evaluated at sensible values."""
        # create engine
        engine = self.create_engine()

        # test criteria
        assert engine.T_0_ratio > 1
        assert engine.p_0_ratio > 1
        assert engine.nozzle_area_ratio > 0
        assert engine.C_th > 0
        assert engine.jet_velocity_ratio > 0
        assert engine.jet_velocity_ratio <= 1
        assert engine.eta_comp > 0
        assert engine.eta_comp <= 1
        assert engine.eta_prop > 0
        assert engine.eta_prop <= 1
        assert engine.eta_overall > 0
        assert engine.eta_overall <= 1
        
        # check for nans
        assert np.any(np.isnan(engine.blade_rows[-1].exit.M)) == False

    def test_engine_stages(self):
        """Test engine creation with custom number of stages."""
        # loop for different numbers of stages
        no_of_stages = 5
        for index in range(no_of_stages):

            # create engine with custom number of stages
            engine = self.create_engine(no_of_stages = index + 1)
        
            # test criteria
            assert len(engine.stages) == index + 1
            assert len(engine.blade_rows) == 2 * (index + 1)

            # check for nans
            assert np.any(np.isnan(engine.blade_rows[-1].exit.M)) == False

    @pytest.mark.parametrize("no_of_stages", [1, 2, 3])
    @pytest.mark.parametrize("altitude", [0, 5000, 10000])
    @pytest.mark.parametrize("flight_speed", [10, 20, 50])
    def test_engine_matrix(self, no_of_stages, altitude, flight_speed):
        engine = self.create_engine(
            no_of_stages = no_of_stages,
            altitude = altitude,
            flight_speed = flight_speed
        )
        Assert.no_nans(engine)
        #assert_performance_realistic(engine)
        #assert_engine_fundamentals(engine)
