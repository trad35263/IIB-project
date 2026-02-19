# import modules
import numpy as np
import matplotlib.pyplot as plt
import wx
import utils
import traceback

# import custom classes
from flight_scenario import Flight_scenario
from engine import Engine
from geometry import Geometry

# mainframe class
class MainFrame(wx.Frame):
    """Used to create the primary container for the GUI."""
    # create list of label-pairs required to create a scenario object
    scenario_input_labels = [
        ["Label", "label"],
        ["Altitude (m)", "altitude"],
        ["Flight Speed (m/s)", "flight_speed"],
        ["Diameter (m)", "diameter"],
        ["Hub-tip Ratio", "hub_tip_ratio"],
        ["Thrust (N)", "thrust"]
    ]

    # create list of scenario label-pairs to display
    scenario_display_labels = (
        scenario_input_labels + [
            ["Mach Number", "M"],
            ["Thrust Coefficient", "C_th"]
        ]
    )
    
    # create list of label-pairs required to create an engine object
    engine_input_labels = [
        ["No. of Stages", "no_of_stages"],
        ["Vortex Exponent", "vortex_exponent"],
        ["Solver Order", "solver_order"],
        ["Stagnation Pressure Loss Coefficient", "Y_p"],
        ["Flow Coefficient", "phi"],
        ["Stage Loading Coefficient", "psi"]
    ]

    # create list of engine label-pairs to display
    engine_display_labels = (
        engine_input_labels + [
            ["Pressure Ratio", "pressure_ratio"],
            ["Nozzle Area Ratio", "nozzle_area_ratio"],
            ["Inlet Mach Number", "M_1"],
            ["Compressor Efficiency", "eta_comp"],
            ["Nozzle Efficiency", "eta_nozz"],
            ["Propulsive Efficiency", "eta_prop"],
            ["Jet Velocity Ratio", "jet_velocity_ratio"]
        ]
    )
    
    # create list of label-pairs required to create a geometry object
    geometry_input_labels = [
        ["Aspect Ratio", "aspect_ratio"],
        ["Diffusion Factor", "diffusion_factor"],
        ["Deviation Constant", "deviation_constant"]
    ]

    # create list of geometry label-pairs to display
    geometry_display_labels = (
        geometry_input_labels + [
        ]
    )
    def __init__(self):
        """Creates instance of the MainFrame class."""
        # create gui and specify title and size
        super().__init__(
            parent = None,
            title = "High Speed Solver",
            size = (1400, 600)
        )

        # store Flight_scenario instances as dictionary
        self.flight_scenarios = {}
        
        # store panel and sizer references for dynamic grid updates
        self.panel = None
        self.geometry_column = None

        # loop over key-value pairs of default input arguments
        for key, values in utils.Defaults.flight_scenarios.items():

            # take default arguments and construct corresponding class
            self.flight_scenarios[key] = Flight_scenario(*values)

        # create panel
        self.panel = wx.Panel(self)

        # create dropdown listing flight scenarios and bind to event function
        self.scenario_dropdown = wx.ComboBox(
            self.panel,
            choices = [str(key) for key in self.flight_scenarios.keys()],
            style = wx.CB_READONLY,
            value = str(next(iter(self.flight_scenarios)))
        )
        self.scenario_dropdown.Bind(wx.EVT_COMBOBOX, self.change_scenario)

        # create dropdown listing associated engines and bind to event function
        self.engine_dropdown = wx.ComboBox(
            self.panel,
            choices = [],
            style = wx.CB_READONLY
        )
        self.engine_dropdown.Bind(wx.EVT_COMBOBOX, self.change_engine)

        # create dropdown listing geometry options and bind to event function
        self.geometry_dropdown = wx.ComboBox(
            self.panel,
            choices = [],
            style = wx.CB_READONLY
        )
        self.geometry_dropdown.Bind(wx.EVT_COMBOBOX, self.change_geometry)

        # create button to add new flight scenarios and bind to event function
        self.add_scenario_button = wx.Button(self.panel, label = "Add Flight Scenario")
        self.add_scenario_button.Bind(wx.EVT_BUTTON, self.add_scenario)

        # create button to add new engines and bind to event function
        self.add_engine_button = wx.Button(self.panel, label = "Create Engine")
        self.add_engine_button.Bind(wx.EVT_BUTTON, self.add_engine)

        # create button to add new geometries and bind to event function
        self.add_geometry_button = wx.Button(self.panel, label = "Add Geometry")
        self.add_geometry_button.Bind(wx.EVT_BUTTON, self.add_geometry)

        # create buttons for each Engine class method
        methods = [name for name, f in Engine.__dict__.items() if callable(f) and "plot" in name]
        buttons = [wx.Button(self.panel, label = method) for method in methods]
        for button in buttons:

            # bind button to event function
            button.Bind(wx.EVT_BUTTON, self.display_plot)
        
        # create export button and bind to event function
        self.export_button = wx.Button(self.panel, label = "Export Engine")
        self.export_button.Bind(wx.EVT_BUTTON, self.export_engine)

        # create empty lists of text boxes with values to be edited
        self.scenario_display_texts = []
        self.engine_display_texts = []
        self.geometry_display_texts = []

        # create a 2-column grid for flight scenario details
        self.scenario_grid = wx.FlexGridSizer(
            rows = len(self.scenario_display_labels), cols = 2, hgap = 10, vgap = 5
        )
        self.scenario_label = self.scenario_dropdown.GetValue()

        # create a 2-column grid for engine details
        self.engine_grid = wx.FlexGridSizer(
            rows = len(self.engine_display_labels), cols = 2, hgap = 10, vgap = 5
        )

        # create a 2-column grid for geometry details
        self.geometry_grid = wx.FlexGridSizer(
            rows = len(self.geometry_display_labels), cols = 2, hgap = 10, vgap = 5
        )

        # loop over label-pairs for scenarios
        for (display, attribute) in self.scenario_display_labels:

            # add text and store value
            self.scenario_grid.Add(wx.StaticText(self.panel, label = display))
            value = getattr(self.flight_scenarios[self.scenario_label], attribute)

            # for string attributes
            if isinstance(value, str):

                # display as is
                text = wx.StaticText(self.panel, label = f"{value}")

            # for numeric attributes
            else:

                # display to 4 significant figures
                text = wx.StaticText(self.panel, label = f"{value:.4g}")

            # store text object and add to grid
            self.scenario_display_texts.append(text)
            self.scenario_grid.Add(text)

        # loop over label-pairs for engines
        for (display, attribute) in self.engine_display_labels:

            # add text and store value
            self.engine_grid.Add(wx.StaticText(self.panel, label = display))
            text = wx.StaticText(self.panel, label = "")

            # store text object and add to grid
            self.engine_display_texts.append(text)
            self.engine_grid.Add(text)

        # loop over label-pairs for geometry
        for (display, attribute) in self.geometry_display_labels:

            # add text and store value
            self.geometry_grid.Add(wx.StaticText(self.panel, label = display))
            text = wx.StaticText(self.panel, label = "")

            # store text object and add to grid
            self.geometry_display_texts.append(text)
            self.geometry_grid.Add(text)

        # create left-hand scenario column
        scenario_column = wx.BoxSizer(wx.VERTICAL)

        # assign button, dropdown and grid to left-hand column
        scenario_column.Add(self.add_scenario_button, 0, wx.ALL | wx.CENTER, 8)
        scenario_column.Add(self.scenario_dropdown, 0, wx.ALL | wx.EXPAND, 8)
        scenario_column.Add(self.scenario_grid, 0, wx.ALL | wx.CENTER, 8)
        scenario_column.AddStretchSpacer()

        # create central engine column
        engine_column = wx.BoxSizer(wx.VERTICAL)

        # assign button, dropdown and grid to centre column
        engine_column.Add(self.add_engine_button, 0, wx.ALL | wx.CENTER, 8)
        engine_column.Add(self.engine_dropdown, 0, wx.ALL | wx.EXPAND, 8)
        engine_column.Add(self.engine_grid, 0, wx.ALL | wx.CENTER, 8)
        engine_column.AddStretchSpacer()

        # create right-hand column
        plot_column = wx.BoxSizer(wx.VERTICAL)

        # assign buttons to right-hand column
        for button in buttons:

            plot_column.Add(button, 0, wx.ALL | wx.EXPAND, 8)
        
        # add stretch spacer to push export button to bottom
        plot_column.AddStretchSpacer()
        
        # add export button at the bottom
        plot_column.Add(self.export_button, 0, wx.ALL | wx.EXPAND, 8)

        # create central geometry column
        self.geometry_column = wx.BoxSizer(wx.VERTICAL)
        geometry_column = self.geometry_column

        # assemble root sizer: add the new geometry column between centre and right
        geometry_column.Add(self.add_geometry_button, 0, wx.ALL | wx.CENTER, 8)
        geometry_column.Add(self.geometry_dropdown, 0, wx.ALL | wx.EXPAND, 8)
        geometry_column.Add(self.geometry_grid, 0, wx.ALL | wx.CENTER, 8)
        geometry_column.AddStretchSpacer()

        # create root sizer
        root = wx.BoxSizer(wx.HORIZONTAL)
        root.Add(scenario_column, proportion = 1, flag = wx.EXPAND | wx.ALL, border = 10)
        root.Add(engine_column, proportion = 1, flag = wx.EXPAND | wx.ALL, border = 10)
        root.Add(geometry_column, proportion = 1, flag = wx.EXPAND | wx.ALL, border = 10)
        root.Add(plot_column, proportion = 1, flag = wx.EXPAND | wx.ALL, border = 10)
        self.panel.SetSizer(root)

        # create colours dictionary
        colours = {}
        colours["dark grey"] = wx.Colour(60, 60, 60)
        colours["light grey"] = wx.Colour(80, 80, 80)
        colours["white"] = wx.Colour(255, 255, 255)
        colours["blue"] = wx.Colour(64, 128, 192)

        # set font
        font = wx.Font(10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, False, "Segoe UI")

        # apply to panel
        self.apply_styling(self.panel, font, colours)

        # show gui
        self.Show()

    def add_scenario(self, event):
        """Executes the addition of an extra dropdown option."""
        # create dialog box
        dialog = AddScenarioDialog(self)

        # if user presses OK
        if dialog.ShowModal() == wx.ID_OK:

            # try-except block
            try:

                # create and store new object
                self.scenario_label = dialog.label.GetValue()
                arguments = (
                    [self.scenario_label]
                    + [float(arg.GetValue()) for arg in dialog.arguments]
                )
                self.flight_scenarios[self.scenario_label] = Flight_scenario(*arguments)

                # refresh dropdown
                self.scenario_dropdown.Append(self.scenario_label)

            # catch non-numeric inputs
            except ValueError:

                # display error
                wx.MessageBox("Please enter a valid numeric input.", "Invalid Input")

        # close dialog box
        dialog.Destroy()

        # set scenario dropdown to the most recent option
        self.scenario_dropdown.SetSelection(self.scenario_dropdown.GetCount() - 1)

        # loop over pairs of scenario labels and text boxes
        for (label, text) in zip(self.scenario_display_labels, self.scenario_display_texts):

            # get new value to display from relevant Flight_scenario instance
            value = getattr(self.flight_scenarios[self.scenario_label], label[1])

            # for string values
            if isinstance(value, str):

                # display as is
                text.SetLabel(f"{value}")

            # for numeric values
            else:

                # display to 4 significant figures
                text.SetLabel(f"{value:.4g}")

        # clear engine selection dropdown
        self.engine_dropdown.Clear()

        # loop over all engine information texts
        for text in self.engine_display_texts:

            # clear labels
            text.SetLabel("")

    def change_scenario(self, event):
        """Executes on each change of the scenario dropdown menu."""
        # get latest label from scenario dropdown
        self.scenario_label = self.scenario_dropdown.GetValue()

        # loop over pairs of scenario labels and text boxes
        for (label, text) in zip(self.scenario_display_labels, self.scenario_display_texts):

            # get new value to display from relevant Flight_scenario instance
            value = getattr(self.flight_scenarios[self.scenario_label], label[1])

            # for string values
            if isinstance(value, str):

                # display as is
                text.SetLabel(f"{value}")

            # for numeric values
            else:

                # display to 4 significant figures
                text.SetLabel(f"{value:.4g}")

        # clear engine selection dropdown
        self.engine_dropdown.Clear()
        engines = self.flight_scenarios[self.scenario_label].engines

        # loop over engines associated with the chosen flight scenario
        for index, engine in enumerate(engines):

            # append engine descriptions to dropdown
            self.engine_dropdown.Append(
                f"[{index}]     Stages: {engine.no_of_stages} | n: {engine.vortex_exponent} | "
                f"N: {engine.solver_order}"
            )

        # if the chosen flight scenario has any associated engines
        if engines:

            # set engine dropdown to the first option
            self.engine_dropdown.SetSelection(0)

            # get latest label from engine dropdown
            self.engine_label = self.engine_dropdown.GetValue()
            self.engine_label = int(self.engine_label.split(']')[0][1:])

            # loop over pairs of ...
            for (label, text) in zip(self.engine_display_labels, self.engine_display_texts):

                # get new value to display from relevant Engine instance
                engine = self.flight_scenarios[self.scenario_label].engines[int(self.engine_label)]
                value = getattr(engine, label[1])

                # for string values
                if isinstance(value, str):

                    # display as is
                    text.SetLabel(f"{value}")

                # for numeric values
                else:

                    # display to 4 significant figures
                    text.SetLabel(f"{value:.4g}")
            
            # rebuild geometry grid for current engine
            self.rebuild_geometry_grid()
            
            # clear geometry dropdown
            self.geometry_dropdown.Clear()
            self.geometry_display_texts.clear()

        # no engines exist
        else:

            # loop over all engine information texts
            for text in self.engine_display_texts:

                # clear labels
                text.SetLabel("")
            
            # rebuild geometry grid for current engine
            self.rebuild_geometry_grid()
            
            # clear geometry dropdown
            self.geometry_dropdown.Clear()
            self.geometry_display_texts.clear()

        # update layout
        self.Layout()

    def add_engine(self, event):
        """Executes the creation of an engine corresponding to the selected flight scenario."""
        # create dialog box
        dialog = AddEngineDialog(self)

        # if user presses OK
        if dialog.ShowModal() == wx.ID_OK:

            # try-except block
            try:

                # create and store new object
                label = self.scenario_dropdown.GetValue()
                arguments = (
                    [self.flight_scenarios[label]]
                    + [float(arg.GetValue()) for arg in dialog.arguments]
                )
                engine = Engine(*arguments)
                self.flight_scenarios[label].engines.append(engine)

                # refresh dropdown
                no_of_stages = int(dialog.arguments[0].GetValue())
                vortex_exponent = float(dialog.arguments[1].GetValue())
                solver_order = int(dialog.arguments[2].GetValue())
                self.engine_dropdown.Append(
                    f"[{len(self.flight_scenarios[label].engines) - 1}]"
                    f"      Stages: {no_of_stages} | n: {vortex_exponent} | N: {solver_order}"
                )

            # catch non-numeric inputs
            except ValueError as error:

                # display error
                wx.MessageBox("Please enter a valid numeric input.", "Invalid Input")
                traceback.print_exc()

        # close dialog box
        dialog.Destroy()

        # set engine dropdown to the most recent option
        self.engine_dropdown.SetSelection(self.engine_dropdown.GetCount() - 1)

        # get latest engine label from engine dropdown
        self.engine_label = self.engine_dropdown.GetValue()
        self.engine_label = int(self.engine_label.split(']')[0][1:])

        # loop over pairs of scenario labels and text boxes
        for (label, text) in zip(self.engine_display_labels, self.engine_display_texts):

            # get new value to display from relevant Engine instance
            value = getattr(engine, label[1])

            # for string values
            if isinstance(value, str):

                # display as is
                text.SetLabel(f"{value}")

            # for numeric values
            else:

                # display to 4 significant figures
                text.SetLabel(f"{value:.4g}")
            
        # rebuild geometry grid for current engine
        self.rebuild_geometry_grid()
        
        # clear geometry dropdown and texts
        self.geometry_dropdown.Clear()
        self.geometry_display_texts.clear()

    def change_engine(self, event):
        """Executes on each change of the engine dropdown menu."""
        # get latest label from engine dropdown
        self.engine_label = self.engine_dropdown.GetValue()
        self.engine_label = int(self.engine_label.split(']')[0][1:])

        # loop over pairs of ...
        for (label, text) in zip(self.engine_display_labels, self.engine_display_texts):

            # get new value to display from relevant Engine instance
            engine = self.flight_scenarios[self.scenario_label].engines[int(self.engine_label)]
            value = getattr(engine, label[1])

            # for string values
            if isinstance(value, str):

                # display as is
                text.SetLabel(f"{value}")

            # for numeric values
            else:

                # display to 4 significant figures
                text.SetLabel(f"{value:.4g}")
            
        # rebuild geometry grid for current engine
        self.rebuild_geometry_grid()
        
        # clear geometry dropdown
        self.geometry_dropdown.Clear()
        self.geometry_display_texts.clear()

    def add_geometry(self, event):
        """Shell for adding a new geometry entry (dialog not implemented)."""
        # create dialog box
        dialog = AddGeometryDialog(self)

        # get latest engine from engine dropdow
        engine = self.flight_scenarios[self.scenario_label].engines[self.engine_label]
        
        # if user presses OK
        if dialog.ShowModal() == wx.ID_OK:

            # try-except block
            try:

                # create and store new object
                aspect_ratio = float(dialog.arguments[0].GetValue())
                diffusion_factor = float(dialog.arguments[1].GetValue())
                deviation_constant = float(dialog.arguments[2].GetValue())
                geometry = Geometry(
                    aspect_ratio, diffusion_factor, deviation_constant
                )
                engine.geometries.append(geometry)
                engine.geometry = geometry
                engine.empirical_design()

                # add new entry to dropdown
                self.geometry_dropdown.Append(
                    f"[{len(engine.geometries) - 1}]        "
                    f"AR: {aspect_ratio} | DF: {diffusion_factor} | m: {deviation_constant}"
                )

            # catch non-numeric inputs
            except ValueError as error:

                # display error
                wx.MessageBox("Please enter a valid numeric input.", "Invalid Input")
                traceback.print_exc()

        # close dialog box
        dialog.Destroy()

        # set geometry dropdown to the most recent option
        self.geometry_dropdown.SetSelection(self.geometry_dropdown.GetCount() - 1)

        # rebuild geometry grid
        self.geometry_label = len(engine.geometries) - 1
        self.rebuild_geometry_grid()

    def change_geometry(self, event):
        """Executes on each change of the geometry dropdown menu."""
        # get latest label from geometry dropdown
        self.geometry_label = int(self.geometry_dropdown.GetValue().split(']')[0][1:])
        engine = self.flight_scenarios[self.scenario_label].engines[self.engine_label]
        geometry = engine.geometries[self.geometry_label]
        engine.geometry = geometry
        engine.empirical_design()
        """for (blade_row, no_of_blades) in zip(engine.blade_rows, geometry.no_of_blades):

            blade_row.no_of_blades = no_of_blades"""
        
        # rebuild geometry grid
        self.rebuild_geometry_grid()

    def rebuild_geometry_grid(self):
        """Dynamically rebuilds the geometry grid based on the number of stages."""
        # remove old grid from sizer
        self.geometry_column.Detach(self.geometry_grid)
        self.geometry_grid.Clear(True)

        # check if an engine has been created yet
        if len(self.flight_scenarios[self.scenario_label].engines) == 0:

            # no engine-specific information to display
            engine = None
            no_of_stages = 0

        else:

            # get current engine instance and extract information
            engine = self.flight_scenarios[self.scenario_label].engines[self.engine_label]
            no_of_stages = len(engine.stages)
        
        # create new geometry grid with updated row count
        self.geometry_grid = wx.FlexGridSizer(
            rows = len(self.geometry_input_labels) + 2 * no_of_stages, cols = 2, hgap = 10, vgap = 5
        )

        # clear geometry display texts
        self.geometry_display_texts = []

        # loop over label-pairs for geometry
        for (display, attribute) in self.geometry_display_labels:

            # add text and store value
            self.geometry_grid.Add(wx.StaticText(self.panel, label = display))
            if engine is not None:

                if len(engine.geometries) == 0:

                    text = wx.StaticText(self.panel, label = "")

                else:

                    text = wx.StaticText(
                        self.panel,
                        label = f"{getattr(engine.geometries[self.geometry_label], attribute)}"
                    )

            else:

                text = wx.StaticText(self.panel, label = "")

            # store text object and add to grid
            self.geometry_display_texts.append(text)
            self.geometry_grid.Add(text)

        # loop over pairs of geometry labels and text boxes
        """for (label, text) in zip(self.geometry_display_labels, self.geometry_display_texts):

            # get new value to display from relevant Engine instance
            if len(engine.geometries) == 0:

                value = ""

            else:

                value = getattr(engine.geometries[-1], label[1])

            # for string values
            if isinstance(value, str):

                # display as is
                text.SetLabel(f"{value}")

            # for numeric values
            else:

                # display to 4 significant figures
                text.SetLabel(f"{value:.4g}")"""
        
        # add rows for each stage
        for index in range(no_of_stages):

            # create grid entry for number of rotor blades
            self.geometry_grid.Add(
                wx.StaticText(self.panel, label = f"Rotor {index + 1} Blades")
            )

            # check if blade counts have already been calculated
            if hasattr(engine.stages[index].rotor, "no_of_blades"):

                # populated text field with blade count
                text = wx.StaticText(self.panel, label = f"{engine.stages[index].rotor.no_of_blades}")
            
            # no blade counts available
            else:
                
                # empty text box
                text = wx.StaticText(self.panel, label = "")
        
            # append to list of texts and add to grid
            self.geometry_display_texts.append(text)
            self.geometry_grid.Add(text)

            # create grid entry for number of stator blades
            self.geometry_grid.Add(
                wx.StaticText(self.panel, label = f"Stator {index + 1} Blades")
            )

            # check if blade counts have already been calculated
            if hasattr(engine.stages[index].stator, "no_of_blades"):

                # populated text field with blade count
                text = wx.StaticText(self.panel, label = f"{engine.stages[index].stator.no_of_blades}")
            
            # no blade counts available
            else:
                
                # empty text box
                text = wx.StaticText(self.panel, label = "")
        
            # append to list of texts and add to grid
            self.geometry_display_texts.append(text)
            self.geometry_grid.Add(text)
        
        # add new grid to sizer at position 2 (after dropdown)
        self.geometry_column.Insert(2, self.geometry_grid, 0, wx.ALL | wx.CENTER, 8)
        
        # refresh layout
        self.panel.Layout()

    def display_plot(self, event):
        """Displays a given plot for the selected Engine class instance."""
        # retrieve button and corresponding method name
        button = event.GetEventObject()
        method = button.GetLabel()

        # get current instance of engine class
        if len(self.flight_scenarios[self.scenario_label].engines) == 0:

            pass

        else:

            # retrieve method and call function
            engine = self.flight_scenarios[self.scenario_label].engines[int(self.engine_label)]
            method = getattr(engine, method)
            method()

        plt.show()
    
    def export_engine(self, event):
        """Exports the current engine to .mat and .json files."""
        # get current instance of engine class
        if len(self.flight_scenarios[self.scenario_label].engines) == 0:

            pass

        else:

            # retrieve engine and call export function
            engine = self.flight_scenarios[self.scenario_label].engines[self.engine_label]
            engine.export()
        
    def apply_styling(self, parent, font, colours):
        """Applies styling to a child object for a given parent object."""
        # wrap in try-except block to ensure gui loads in spite of style errors
        try:
            
            # apply background to frame and main panel
            self.SetBackgroundColour(colours["dark grey"])
            self.panel.SetBackgroundColour(colours["dark grey"])

        except Exception:

            pass

        # loop over all children
        for child in parent.GetChildren():

            # wrap in try-except block to ensure gui loads in spite of style errors
            try:

                # if child instance is a button
                if isinstance(child, wx.Button):

                    # set background and border colour
                    child.SetBackgroundColour(colours["blue"])
                    child.SetForegroundColour(colours["white"])

                    # set font and increase vertical padding
                    child.SetFont(font)
                    child.SetMinSize((-1, 32))

                # if child instance is a text box
                elif isinstance(child, wx.StaticText):

                    # set background and border colour and font
                    child.SetBackgroundColour(colours["dark grey"])
                    child.SetForegroundColour(colours["white"])
                    child.SetFont(font)

                # if child instance is a drop-down menu
                elif isinstance(child, wx.ComboBox) or isinstance(child, wx.TextCtrl):
                    
                    # set background and border colour and font
                    child.SetBackgroundColour(colours["light grey"])
                    child.SetForegroundColour(colours["white"])
                    child.SetFont(font)

            except Exception:

                pass

            # recurse into nested containers
            if hasattr(child, "GetChildren"):

                # apply styling function
                self.apply_styling(child, font, colours)

# create AddScenarioDialog class
class AddScenarioDialog(wx.Dialog):
    """Creates a dialog box for inputting arguments for additional Flight_scenario classes."""
    def __init__(self, parent):
        """Creates an instance of the AddScenarioDialog class."""
        # create dialog box
        super().__init__(
            parent,
            title = "Add Flight Scenario",
            size = (1000, 300)
        )
        panel = wx.Panel(self)

        # loop over all input label-pairs
        for input_label in parent.scenario_input_labels:

            # set the relevant attribute to a newly created text box
            setattr(self, input_label[1], wx.TextCtrl(panel, value = f"{getattr(utils.Defaults, input_label[1])}"))

        # layout in grid form
        grid = wx.FlexGridSizer(
            rows = len(parent.scenario_input_labels), cols = 2, hgap = 10, vgap = 8
        )
        grid.AddMany([
            item for input_label in parent.scenario_input_labels
            for item in (
                wx.StaticText(panel, label = input_label[0]),
                getattr(self, input_label[1]),
            )
        ])

        # panel-level sizer (for the form)
        panel_sizer = wx.BoxSizer(wx.VERTICAL)
        panel_sizer.Add(grid, 1, wx.ALL | wx.EXPAND, 15)
        panel.SetSizer(panel_sizer)

        # dialog-level buttons
        buttons = self.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL)

        # dialog root sizer
        dialog_sizer = wx.BoxSizer(wx.VERTICAL)
        dialog_sizer.Add(panel, 1, wx.EXPAND | wx.ALL, 5)
        dialog_sizer.Add(buttons, 0, wx.EXPAND | wx.ALL, 10)
        self.SetSizerAndFit(dialog_sizer)

        # store as ordered list of arguments
        self.arguments = [
            getattr(self, input_label[1]) for input_label in parent.scenario_input_labels[1:]
        ]

# create AddEngineDialog class
class AddEngineDialog(wx.Dialog):
    """Creates a dialog box for creating an additional Engine class."""
    def __init__(self, parent):
        """Creates an instance of the AddEngineDialog class."""
        # create dialog box
        super().__init__(
            parent,
            title = "Create Engine",
            size = (1000, 300)
        )
        panel = wx.Panel(self)

        # loop over all input label-pairs
        for input_label in parent.engine_input_labels:

            # set the relevant attribute to a newly created text box
            setattr(self, input_label[1], wx.TextCtrl(panel, value = f"{getattr(utils.Defaults, input_label[1])}"))

        # layout in grid form
        grid = wx.FlexGridSizer(
            rows = len(parent.engine_input_labels), cols = 2, hgap = 10, vgap = 8
        )
        grid.AddMany([
            item
            for input_label in parent.engine_input_labels
            for item in (
                wx.StaticText(panel, label = input_label[0]),
                getattr(self, input_label[1]),
            )
        ])

        # panel-level sizer (for the form)
        panel_sizer = wx.BoxSizer(wx.VERTICAL)
        panel_sizer.Add(grid, 1, wx.ALL | wx.EXPAND, 15)
        panel.SetSizer(panel_sizer)

        # Dialog-level buttons
        buttons = self.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL)

        # Dialog root sizer
        dialog_sizer = wx.BoxSizer(wx.VERTICAL)
        dialog_sizer.Add(panel, 1, wx.EXPAND | wx.ALL, 5)
        dialog_sizer.Add(buttons, 0, wx.EXPAND | wx.ALL, 10)
        self.SetSizerAndFit(dialog_sizer)

        # store as ordered list of arguments
        self.arguments = [
            getattr(self, input_label[1]) for input_label in parent.engine_input_labels
        ]

# create AddGeometryDialog class
class AddGeometryDialog(wx.Dialog):
    """Creates a dialog box for creating or editing geometry entries."""
    def __init__(self, parent):
        """Creates an instance of the AddGeometryDialog class."""
        super().__init__(
            parent,
            title = "Add Geometry",
            size = (800, 250)
        )
        panel = wx.Panel(self)

        # loop over all input label-pairs
        for input_label in parent.geometry_input_labels:

            # set the relevant attribute to a newly created text box
            default = getattr(utils.Defaults, input_label[1], "")
            setattr(self, input_label[1], wx.TextCtrl(panel, value = f"{default}"))

        # layout in grid form
        grid = wx.FlexGridSizer(
            rows = len(parent.geometry_input_labels), cols = 2, hgap = 10, vgap = 8
        )
        grid.AddMany([
            item for input_label in parent.geometry_input_labels
            for item in (
                wx.StaticText(panel, label = input_label[0]),
                getattr(self, input_label[1]),
            )
        ])

        # panel-level sizer (for the form)
        panel_sizer = wx.BoxSizer(wx.VERTICAL)
        panel_sizer.Add(grid, 1, wx.ALL | wx.EXPAND, 15)
        panel.SetSizer(panel_sizer)

        # dialog-level buttons
        buttons = self.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL)

        # dialog root sizer
        dialog_sizer = wx.BoxSizer(wx.VERTICAL)
        dialog_sizer.Add(panel, 1, wx.EXPAND | wx.ALL, 5)
        dialog_sizer.Add(buttons, 0, wx.EXPAND | wx.ALL, 10)
        self.SetSizerAndFit(dialog_sizer)

        # store as ordered list of arguments
        self.arguments = [getattr(self, input_label[1]) for input_label in parent.geometry_input_labels]

# main function
def main():
    """Runs on script execution."""
    # create gui
    app = wx.App(False)
    frame = MainFrame()
    app.MainLoop()

# run on script execution
if __name__ == "__main__":

    # run main function
    main()
