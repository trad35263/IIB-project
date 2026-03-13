# import modules
import numpy as np
import matplotlib.pyplot as plt
import wx
import utils
import traceback
import copy

# import custom classes
from flight_scenario import Flight_scenario
from engine import Engine

# create DataContainer class
class DataContainer:
    """Stores information associated with a certain aspect of the engine design."""
    def __init__(self, label, button_text, panel):
        """Creates an instance of the DataContainer class."""
        # store input variables
        self.label = label
        self.button_text = button_text
        self.panel = panel
        
        # create button and bind to event function
        self.button = wx.Button(self.panel, label = f"{self.button_text}")

        # get input and output labels from utils.Labels class
        self.input_labels = copy.deepcopy(getattr(utils.Labels, f"{self.label}_input_labels"))
        self.output_labels = copy.deepcopy(getattr(utils.Labels, f"{self.label}_output_labels"))
        self.labels = [self.input_labels, self.output_labels]

        # extra labels are available
        if hasattr(utils.Labels, f"{label}_extra_labels"):

            # store extra labels
            self.extra_labels = copy.deepcopy(getattr(utils.Labels, f"{self.label}_extra_labels"))
            self.labels.append(self.extra_labels)

        # create list of grids
        self.grids = []

        # loop over all stored labels
        for label in self.labels:

            # create grid of values to display
            grid = wx.FlexGridSizer(
                rows = len(label), cols = 2, hgap = 10, vgap = 4
            )
            grid.AddGrowableCol(1)
            self.grids.append(grid)

        # create a dropdown menu
        self.dropdown = wx.ComboBox(
            self.panel,
            choices = [],
            style = wx.CB_READONLY
        )

        # set information source to None
        self.source = None

    def populate_column(self, column = None):
        """Appends associated elements to a GUI column."""
        # check if no column argument has been provided
        if column == None:

            # create column
            self.column = wx.BoxSizer(wx.VERTICAL)

        # column has been provided
        else:

            # use existing column
            self.column = column

        # add button and dropdown to column
        self.column.Add(self.button, 0, wx.ALL | wx.CENTER, 8)
        self.column.Add(self.dropdown, 0, wx.ALL | wx.EXPAND, 8)

        # loop over all grids
        for index, grid in enumerate(self.grids):

            if index > 0:
        
                divider = wx.StaticLine(self.panel, style = wx.LI_HORIZONTAL)
                self.column.Add(divider, 0, wx.ALL | wx.EXPAND, 8)

            self.column.Add(grid, 0, wx.ALL | wx.EXPAND, 8)

    def refresh_grid(self):
        """Refreshes the grid for a given engine design aspect."""
        # handle both input and output grids
        for (label, grid) in zip(self.labels, self.grids):

            # clear existing grids and modify number of rows
            grid.Clear(True)
            grid.SetRows(len(label))

            # loop over label-pairs for scenario inputs
            for (key, attribute) in label:

                # add left-hand text entry
                grid.Add(wx.StaticText(self.panel, label = key), flag=wx.ALIGN_LEFT)

                # source is a dictionary
                if isinstance(self.source, dict):
                    
                    # check if relevant attribute exists
                    if attribute not in self.source:
                        
                        # display nothing
                        grid.Add(wx.StaticText(self.panel, label = ""))
                        continue
                    
                    # store value to display
                    value = self.source[attribute]

                # source is a class
                else:

                    # check if relevant attribute exists
                    if not hasattr(self.source, attribute):

                        # display nothing
                        grid.Add(wx.StaticText(self.panel, label = ""))
                        continue

                    # store value to display
                    value = getattr(self.source, attribute)

                # for string attributes
                if isinstance(value, str):

                    # display as is
                    text = wx.StaticText(self.panel, label = f"{value}")

                # for list attributes
                elif isinstance(value, list):

                    # display first entry to 4 significant figures
                    text = wx.StaticText(self.panel, label = f"{value[0]:.4g}")

                # for numeric attributes
                else:

                    # display to 4 significant figures
                    text = wx.StaticText(self.panel, label = f"{value:.4g}")

                # add text object to grid
                grid.Add(text, flag=wx.ALIGN_RIGHT | wx.EXPAND)

# create MainFrame class
class MainFrame(wx.Frame):
    """Used to create the primary container for the GUI."""
    def __init__(self):
        """Creates instance of the MainFrame class."""
        # create gui and specify title and size
        super().__init__(
            parent = None,
            title = "High Speed Solver",
            size = (1400, 800)
        )

        # store Flight_scenario instances as dictionary
        self.flight_scenarios = {}

        # loop over key-value pairs of default input arguments
        for key, values in utils.Defaults.flight_scenarios.items():

            # take default arguments and construct flight scenarios
            self.flight_scenarios[key] = Flight_scenario(*values)

        # create panel
        self.panel = wx.Panel(self)

        # create containers for the different engine design aspects
        self.scenario = DataContainer("scenario", "Add Flight Scenario", self.panel)
        self.engine = DataContainer("engine", "Create Engine", self.panel)
        self.geometry = DataContainer("geometry", "Add Geometry", self.panel)
        self.off_design = DataContainer("off_design", "Off-design", self.panel)
        self.thickness = DataContainer("thickness", "Add Thickness", self.panel)
        data_containers = [
            self.scenario, self.engine,
            self.geometry, self.off_design,
            self.thickness
        ]

        # loop for each kind of data container
        for data in data_containers:

            # bind button and dropdown to corresponding event function
            data.button.Bind(wx.EVT_BUTTON, getattr(self, f"add_{data.label}"))
            data.dropdown.Bind(wx.EVT_COMBOBOX, getattr(self, f"change_{data.label}"))

            # populate column
            data.populate_column()

        # loop over all default flight scenario labels
        for key in self.flight_scenarios.keys():
            
            # append as flight scenario dropdown option
            self.scenario.dropdown.Append(str(key))
        
        # set default flight scenario to first entry stored in dictionary
        self.scenario.label = str(next(iter(self.flight_scenarios)))
        self.scenario.dropdown.SetValue(self.scenario.label)
        self.scenario.source = self.flight_scenarios[self.scenario.label]

        # create buttons for each Engine plotting function
        plot_methods = [name for name, f in Engine.__dict__.items() if callable(f) and "plot" in name]
        plot_buttons = [wx.Button(self.panel, label = method) for method in plot_methods]
        for button in plot_buttons:

            # bind button to event function
            button.Bind(wx.EVT_BUTTON, self.display_plot)
        
        # create export button and bind to event function
        self.export_button = wx.Button(self.panel, label = "Export Engine")
        self.export_button.Bind(wx.EVT_BUTTON, self.export_engine)
        
        # add stretch spacer to geometry column (after off_design is added)
        #self.geometry.column.AddStretchSpacer()

        # toggle slider at the bottom of the scenario column
        self.toggle_slider = wx.Slider(
            self.panel, value = 0, minValue = 0, maxValue = 1, style = wx.SL_HORIZONTAL
        )
        self.toggle_slider.Bind(wx.EVT_SLIDER, self.toggle)

        # place slider and label in a horizontal row where the slider expands
        slider_row = wx.BoxSizer(wx.HORIZONTAL)
        slider_row.Add(self.toggle_slider, 1, wx.ALL | wx.EXPAND, 0)
        debug_label = wx.StaticText(self.panel, label = "Debug Mode")
        slider_row.Add(debug_label, 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL, 8)
        self.scenario.column.Add(slider_row, 0, wx.ALL | wx.EXPAND, 8)

        # add stretch spacer to push buttons to bottom
        self.scenario.column.AddStretchSpacer()

        # loop for each button
        for button in plot_buttons:

            # add plot buttons to scenario column
            self.scenario.column.Add(button, 0, wx.ALL | wx.EXPAND, 8)

        # add export button to scenario column
        self.scenario.column.Add(self.export_button, 0, wx.ALL | wx.EXPAND, 8)
        
        # refresh grids
        self.scenario.refresh_grid()
        self.engine.refresh_grid()
        self.geometry.refresh_grid()
        self.off_design.refresh_grid()

        # create root 
        root = wx.BoxSizer(wx.HORIZONTAL)

        # loop for each data container
        for data in data_containers:

            # add column to root
            root.Add(data.column, proportion = 1, flag = wx.EXPAND | wx.ALL, border = 10)

        # set root sizer
        self.panel.SetSizer(root)

        # create colours dictionary
        self.colours = {}
        self.colours["dark grey"] = wx.Colour(32, 32, 32)
        self.colours["light grey"] = wx.Colour(96, 96, 96)
        self.colours["white"] = wx.Colour(255, 255, 255)
        self.colours["blue"] = wx.Colour(96, 160, 255)

        # set font
        self.font = wx.Font(10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, False, "Segoe UI")

        # apply to panel
        self.apply_styling(self.panel)

        # show gui
        self.Show()

    def add_scenario(self, event):
        """Executes the addition of an extra dropdown option."""
        # create dialog box
        dialog = DialogBox(self, "Add Flight Scenario", self.scenario.input_labels)

        # if user presses OK
        if dialog.ShowModal() == wx.ID_OK:

            # try-except block
            try:

                # create and store new object
                self.scenario.label = dialog.arguments[0].GetValue()
                arguments = (
                    [self.scenario.label]
                    + [float(arg.GetValue()) for arg in dialog.arguments[1:]]
                )
                self.scenario.source = Flight_scenario(*arguments)
                self.flight_scenarios[self.scenario.label] = self.scenario.source
                self.engine.source = None
                self.geometry.source = None
                self.off_design.source = None

                # refresh dropdown
                self.scenario.dropdown.Append(self.scenario.label)  

                # set scenario dropdown to the most recent option
                self.scenario.dropdown.SetSelection(self.scenario.dropdown.GetCount() - 1)

            # catch non-numeric inputs
            except ValueError:

                # display error
                wx.MessageBox("Please enter a valid numeric input.", "Invalid Input")

        # close dialog box
        dialog.Destroy()
        
        # refresh all grids
        self.panel.Freeze()
        self.scenario.refresh_grid()
        self.engine.refresh_grid()
        self.geometry.refresh_grid()
        self.off_design.refresh_grid()

        # clear all dropdowns
        self.engine.dropdown.Clear()
        self.geometry.dropdown.Clear()
        self.off_design.dropdown.Clear()

        # re-apply styling and refresh layout
        self.apply_styling(self.panel)
        self.panel.Thaw()

    def change_scenario(self, event):
        """Executes on each change of the scenario dropdown menu."""
        # get latest label from scenario dropdown
        self.scenario.label = self.scenario.dropdown.GetValue()
        self.scenario.source = self.flight_scenarios[f"{self.scenario.label}"]

        # flight scenario has no stored engine
        if not hasattr(self.scenario.source, "engine"):

            # set sources to None
            self.engine.source = None
            self.geometry.source = None
            self.off_design.source = None

        # engine exists
        else:

            # set current engine
            self.engine.source = self.scenario.source.engines[0]

            # engine has no stored geometry
            if not hasattr(self.engine.source, "geometry"):

                # set sources to None
                self.geometry.source = None
                self.off_design.source = None

            # geometry exists
            else:

                # set current geometry
                self.geometry.source = self.engine.source.geometries[0]

                # engine has no stored off-design
                if not hasattr(self.engine.source, "off_design"):

                    # set source to None
                    self.off_design.source = None

                # off-design exists
                else:

                    # set current off-design
                    self.off_design.source = self.engine.source.off_designs[0]

        # clear all dropdowns
        self.engine.dropdown.Clear()
        self.geometry.dropdown.Clear()
        self.off_design.dropdown.Clear()

        # loop over engines associated with the chosen flight scenario
        for index, engine in enumerate(self.scenario.source.engines):

            # append engine descriptions to dropdown
            self.engine.dropdown.Append(
                f"[{index}]     Stages: {self.engine.source.no_of_stages} | "
                f"n: {self.engine.source.vortex_exponent} | phi: {self.engine.source.phi[0]} | "
                f"psi: {self.engine.source.psi[0]}"
            )

        if self.engine.source != None:

            # loop over engines associated with the chosen flight scenario
            for index, geometry in enumerate(self.engine.source.geometries):

                # append engine descriptions to dropdown
                self.geometry.dropdown.Append(
                    f"[{len(self.engine.source.geometries) - 1}]        AR: {geometry['aspect_ratio']}"
                    f" | DF: {geometry['diffusion_factor']} | p: {geometry['design_parameter']}"
                )

                # loop over engines associated with the chosen flight scenario
                for index, off_design in enumerate(geometry["off_designs"]):

                    # append engine descriptions to dropdown
                    self.off_design.dropdown.Append(
                        f"[{len(engine.off_designs) - 1}]        "
                        f"phi_min: {off_design['phi_min']} | phi_max: {off_design['phi_max']}"
                    )
        
        # refresh all grids
        self.panel.Freeze()
        self.scenario.refresh_grid()
        self.engine.refresh_grid()
        self.geometry.refresh_grid()
        self.off_design.refresh_grid()

        # set dropdown selections to first option
        self.engine.dropdown.SetSelection(0)
        self.geometry.dropdown.SetSelection(0)
        self.off_design.dropdown.SetSelection(0)

        # re-apply styling and refresh layout
        self.apply_styling(self.panel)
        self.panel.Layout()
        self.panel.Thaw()

    def add_engine(self, event):
        """Creates an engine corresponding to the selected flight scenario."""
        # create dialog box
        dialog = DialogBox(self, "Add Engine", self.engine.input_labels)

        # if user presses OK
        if dialog.ShowModal() == wx.ID_OK:

            # try-except block
            try:

                # create and store new engine
                arguments = (
                    [self.scenario.source]
                    + [float(arg.GetValue()) for arg in dialog.arguments]
                )
                engine = Engine(*arguments)
                self.scenario.source.engines.append(engine)
                self.scenario.source.engine = engine
                self.engine.source = engine
                self.geometry.source = None
                self.off_design.source = None

                # add new entry to dropdown
                self.engine.dropdown.Append(
                    f"[{len(self.scenario.source.engines) - 1}]     Stages: {self.engine.source.no_of_stages} | "
                    f"n: {self.engine.source.vortex_exponent} | phi: {self.engine.source.phi[0]} | "
                    f"psi: {self.engine.source.psi[0]}"
                )

                # set engine dropdown to the most recent option
                self.engine.dropdown.SetSelection(self.engine.dropdown.GetCount() - 1)

            # catch non-numeric inputs
            except ValueError as error:

                # display error
                wx.MessageBox("Please enter a valid numeric input.", "Invalid Input")
                traceback.print_exc()

        # close dialog box
        dialog.Destroy()

        # update engine extra grid for number of blade rows
        self.engine.labels[-1] = copy.deepcopy(utils.Labels.engine_extra_labels)
        for index, stage in enumerate(self.engine.source.stages):

            self.engine.labels[-1].append(
                [f"Rotor {index + 1} rpm", f"rotor_{index + 1}_rpm"]
            )
            self.engine.labels[-1].append(
                [f"Rotor {index + 1} power (W)", f"rotor_{index + 1}_power"]
            )

        # update geometry output grid for number of blade rows
        self.geometry.labels[1] = copy.deepcopy(utils.Labels.geometry_output_labels)
        for index, stage in enumerate(self.engine.source.stages):

            self.geometry.labels[1].append(
                [f"Rotor {index + 1} blades", f"no_of_blades{2 * index}"]
            )
            self.geometry.labels[1].append(
                [f"Rotor {index + 1} min. chord (mm)", f"rotor_{index + 1}_min_chord"]
            )
            self.geometry.labels[1].append(
                [f"Rotor {index + 1} max. chord (mm)", f"rotor_{index + 1}_max_chord"]
            )
            self.geometry.labels[1].append(
                [f"Stator {index + 1} blades", f"no_of_blades{2 * index + 1}"]
            )
            self.geometry.labels[1].append(
                [f"Stator {index + 1} min. chord (mm)", f"stator_{index + 1}_min_chord"]
            )
            self.geometry.labels[1].append(
                [f"Stator {index + 1} max. chord (mm)", f"stator_{index + 1}_max_chord"]
            )
        
        # refresh all grids
        self.panel.Freeze()
        self.engine.refresh_grid()
        self.geometry.refresh_grid()
        self.off_design.refresh_grid()

        # clear all dropdowns
        self.geometry.dropdown.Clear()
        self.off_design.dropdown.Clear()

        # re-apply styling and refresh layout
        self.apply_styling(self.panel)
        self.panel.Thaw()

    def change_engine(self, event):
        """Executes on each change of the engine dropdown menu."""
        # get latest label from engine dropdown
        self.engine.label = self.engine.dropdown.GetValue()
        self.engine.label = int(self.engine.label.split(']')[0][1:])

        # find and store relevant engine
        engine = self.scenario.source.engines[self.engine.label]
        self.engine.source = engine
        self.scenario.source.engine = engine

        # engine has no stored geometry
        if not hasattr(self.engine.source, "geometry"):

            # set sources to None
            self.geometry.source = None
            self.off_design.source = None

        # geometry exists
        else:

            # set current geometry
            self.geometry.source = self.engine.source.geometries[0]

            # engine has no stored off-design
            if not hasattr(self.engine.source, "off_design"):

                # set source to None
                self.off_design.source = None

            # off-design exists
            else:

                # set current off-design
                self.off_design.source = self.geometry.source["off_designs"][0]

        # clear all dropdowns
        self.geometry.dropdown.Clear()
        self.off_design.dropdown.Clear()

        # loop over geometries associated with the chosen engine
        for index, geometry in enumerate(self.engine.source.geometries):

            # append engine descriptions to dropdown
            self.geometry.dropdown.Append(
                f"[{len(self.engine.source.geometries) - 1}]        AR: {geometry['aspect_ratio']} | "
                f"DF: {geometry['diffusion_factor']} | p: {geometry['design_parameter']}"
            )

        if self.geometry.source != None:

            # loop over geometries associated with the chosen engine
            for index, off_design in enumerate(self.geometry.source["off_designs"]):

                # append engine descriptions to dropdown
                self.off_design.dropdown.Append(
                    f"[{len(self.geometry.source['off_designs']) - 1}]        "
                    f"phi_min: {off_design['phi_min']} | phi_max: {off_design['phi_max']}"
                )

        # update engine extra grid for number of blade rows
        self.engine.labels[-1] = copy.deepcopy(utils.Labels.engine_extra_labels)
        for index, stage in enumerate(self.engine.source.stages):

            self.engine.labels[-1].append(
                [f"Rotor {index + 1} rpm", f"rotor_{index + 1}_rpm"]
            )
            self.engine.labels[-1].append(
                [f"Rotor {index + 1} power (W)", f"rotor_{index + 1}_power"]
            )

        # update geometry output grid for number of blade rows
        self.geometry.labels[1] = copy.deepcopy(utils.Labels.geometry_output_labels)
        for index, stage in enumerate(self.engine.source.stages):

            self.geometry.labels[1].append(
                [f"Rotor {index + 1} blades", f"no_of_blades{2 * index}"]
            )
            self.geometry.labels[1].append(
                [f"Rotor {index + 1} min. chord (mm)", f"rotor_{index + 1}_min_chord"]
            )
            self.geometry.labels[1].append(
                [f"Rotor {index + 1} max. chord (mm)", f"rotor_{index + 1}_max_chord"]
            )
            self.geometry.labels[1].append(
                [f"Stator {index + 1} blades", f"no_of_blades{2 * index + 1}"]
            )
            self.geometry.labels[1].append(
                [f"Stator {index + 1} min. chord (mm)", f"stator_{index + 1}_min_chord"]
            )
            self.geometry.labels[1].append(
                [f"Stator {index + 1} max. chord (mm)", f"stator_{index + 1}_max_chord"]
            )
        
        # refresh all grids
        self.panel.Freeze()
        self.engine.refresh_grid()
        self.geometry.refresh_grid()
        self.off_design.refresh_grid()

        # set dropdowns to first option
        self.geometry.dropdown.SetSelection(0)
        self.off_design.dropdown.SetSelection(0)

        # re-apply styling and refresh layout
        self.apply_styling(self.panel)
        self.panel.Thaw()

    def add_geometry(self, event):
        """Adds a new engine geometry input."""
        # check if an engine has already been created
        if len(self.scenario.source.engines) == 0:

            return
        
        # create dialog box
        dialog = AddGeometryDialog(self)
        
        # if user presses OK
        if dialog.ShowModal() == wx.ID_OK:

            # try-except block
            try:

                # create and store new object
                aspect_ratio = float(dialog.arguments[0].GetValue())
                diffusion_factor = float(dialog.arguments[1].GetValue())
                design_parameter = float(dialog.arguments[2].GetValue())# / 100
                self.geometry.source = {
                    "aspect_ratio": aspect_ratio,
                    "diffusion_factor": diffusion_factor,
                    "design_parameter": design_parameter,
                    "off_designs": []
                }
                self.engine.source.geometries.append(self.geometry.source)
                self.engine.source.geometry = self.geometry.source
                self.engine.source.empirical_design()
                self.off_design.source = None

                # add new entry to dropdown
                self.geometry.dropdown.Append(
                    f"[{len(self.engine.source.geometries) - 1}]        "
                    f"AR: {aspect_ratio} | DF: {diffusion_factor} | p: {design_parameter}"
                )

                # set geometry dropdown to the most recent option
                self.geometry.dropdown.SetSelection(self.geometry.dropdown.GetCount() - 1)

            # catch non-numeric inputs
            except ValueError as error:

                # display error
                wx.MessageBox("Please enter a valid numeric input.", "Invalid Input")
                traceback.print_exc()

        # close dialog box
        dialog.Destroy()
        
        # refresh all grids
        self.panel.Freeze()
        self.geometry.refresh_grid()
        self.off_design.refresh_grid()

        # clear all dropdowns
        self.off_design.dropdown.Clear()

        # re-apply styling and refresh layout
        self.apply_styling(self.panel)
        self.panel.Thaw()

    def change_geometry(self, event):
        """Executes on each change of the geometry dropdown menu."""
        # get latest label from engine dropdown
        self.geometry.label = self.geometry.dropdown.GetValue()
        self.geometry.label = int(self.geometry.label.split(']')[0][1:])

        # find and store relevant engine
        geometry = self.engine.source.geometries[self.geometry.label]
        self.geometry.source = geometry
        self.engine.source.geometry = geometry

        # engine has no stored off-design
        if not hasattr(self.geometry.source, "off_designs"):

            # set source to None
            self.off_design.source = None

        # off-design exists
        else:

            # set current off-design
            self.off_design.source = self.geometry.source["off_designs"][0]

        # clear dropdown
        self.off_design.dropdown.Clear()

        # loop over geometries associated with the chosen engine
        for index, off_design in enumerate(self.geometry.source["off_designs"]):

            # append engine descriptions to dropdown
            self.off_design.dropdown.Append(
                f"[{len(self.geometry.source['off_designs']) - 1}]        "
                f"phi_min: {off_design['phi_min']} | phi_max: {off_design['phi_max']}"
            )
        
        # refresh all grids
        self.panel.Freeze()
        self.geometry.refresh_grid()
        self.off_design.refresh_grid()

        # set dropdowns to first option
        self.off_design.dropdown.SetSelection(0)

        # re-apply styling and refresh layout
        self.apply_styling(self.panel)
        self.panel.Thaw()

    def add_off_design(self, event):
        """Specifies an off-design scenario to calculate."""
        # no geometry has been created for the engine
        if self.geometry.source == None:

            # do nothing
            return
        
        # create dialog box
        dialog = DialogBox(self, "Add Off-design", self.off_design.input_labels)
        
        # if user presses OK
        if dialog.ShowModal() == wx.ID_OK:

            # try-except block
            try:

                # create and store new object
                phi_min = float(dialog.arguments[0].GetValue())
                phi_max = float(dialog.arguments[1].GetValue())
                self.off_design.source = {
                    "phi_min": phi_min,
                    "phi_max": phi_max
                }
                self.geometry.source["off_designs"].append(self.off_design.source)
                self.geometry.source["off_design"] = self.off_design.source
                self.engine.source.calculate_off_design()

                # add new entry to dropdown
                self.off_design.dropdown.Append(
                    f"[{len(self.geometry.source['off_designs']) - 1}]        "
                    f"phi_min: {phi_min} | phi_max: {phi_max}"
                )

            # catch non-numeric inputs
            except ValueError as error:

                # display error
                wx.MessageBox("Please enter a valid numeric input.", "Invalid Input")
                traceback.print_exc()

        # close dialog box
        dialog.Destroy()

        # set off design dropdown to the most recent option
        self.off_design.dropdown.SetSelection(self.off_design.dropdown.GetCount() - 1)
        
        # refresh all grids
        self.panel.Freeze()
        self.off_design.refresh_grid()

        # re-apply styling and refresh layout
        self.apply_styling(self.panel)
        self.panel.Thaw()

    def change_off_design(self, event):
        """Executes on each change of the off-design dropdown menu."""
        # get latest label from off-design dropdown
        self.off_design.label = int(self.off_design.dropdown.GetValue().split(']')[0][1:])

        # find and store relevant off-design
        self.off_design.source = self.geometry.source["off_designs"][self.off_design.label]
        #self.geometry.source.off_design = off_design
        
        # refresh grid
        self.panel.Freeze()
        self.off_design.refresh_grid()

        # re-apply styling and refresh layout
        self.apply_styling(self.panel)
        self.panel.Thaw()

    def add_thickness(self, event):
        pass

    def change_thickness(self, event):
        pass

    def display_plot(self, event):
        """Displays a given plot for the selected Engine class instance."""
        # retrieve button and corresponding method name
        button = event.GetEventObject()
        method = button.GetLabel()

        # no engines have been created yet
        if len(self.scenario.source.engines) == 0:

            pass

        # an engine exists
        else:

            # retrieve method and call function
            method = getattr(self.engine.source, method)
            method()

        # show all plots
        plt.show()
    
    def export_engine(self, event):
        """Exports the current engine to .mat and .json files."""
        # get current instance of engine class
        if len(self.flight_scenarios[self.scenario.label].engines) == 0:

            pass

        else:

            # retrieve engine and call export function
            self.engine.source.export()

    def toggle(self, event):
        """Toggles debug mode."""
        # toggle debug mode inside the Defaults class
        if utils.Defaults.debug == False:

            utils.Defaults.debug = True
        
        else:

            utils.Defaults.debug = False

        print(f"{utils.Colours.PURPLE}Debug mode: {utils.Defaults.debug}{utils.Colours.END}")

    def apply_styling(self, parent):
        """Applies styling to a child object for a given parent object."""
        # store font and colour dictionaries
        font = self.font
        colours = self.colours

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

                # if child instance is a slider
                elif isinstance(child, wx.Slider):

                    # set background and border colour and increase height
                    try:
                        child.SetBackgroundColour(colours["dark grey"])
                        child.SetForegroundColour(colours["white"])
                        child.SetMinSize((-1, 36))
                    except Exception:
                        pass

            except Exception:

                pass

            # recurse into nested containers
            if hasattr(child, "GetChildren"):

                # apply styling function
                self.apply_styling(child)

        # update panel
        self.panel.Layout()

# DialogBox class
class DialogBox(wx.Dialog):
    """Creates a dialog box for inputting arguments as part of the engine design process."""
    def __init__(self, parent, title, input_labels):
        """Creates an instance of the DialogBox class."""
        # create dialog box
        super().__init__(
            parent,
            title = f"{title}",
            size = (1000, 300)
        )
        panel = wx.Panel(self)

        # loop over all input label-pairs
        for input_label in input_labels:

            # set the relevant attribute to a newly created text box
            setattr(self, input_label[1], wx.TextCtrl(panel, value = f"{getattr(utils.Defaults, input_label[1])}"))

        # layout in grid form
        grid = wx.FlexGridSizer(
            rows = len(input_labels), cols = 2, hgap = 10, vgap = 8
        )
        grid.AddMany([
            item for input_label in input_labels
            for item in (
                wx.StaticText(panel, label = input_label[0]),
                getattr(self, input_label[1]),
            )
        ])

        # panel-level sizer (for the form)
        panel_sizer = wx.BoxSizer(wx.VERTICAL)
        panel_sizer.Add(grid, 1, wx.ALL | wx.EXPAND, 15)
        panel.SetSizer(panel_sizer)

        # apply mainframe styling to this dialog (background + children)
        self.SetBackgroundColour(parent.colours["dark grey"])
        panel.SetBackgroundColour(parent.colours["dark grey"])
        parent.apply_styling(self)

        # dialog-level buttons
        buttons = self.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL)

        # dialog root sizer
        dialog_sizer = wx.BoxSizer(wx.VERTICAL)
        dialog_sizer.Add(panel, 1, wx.EXPAND | wx.ALL, 5)
        dialog_sizer.Add(buttons, 0, wx.EXPAND | wx.ALL, 10)
        self.SetSizerAndFit(dialog_sizer)

        # store as list of arguments
        self.arguments = [
            getattr(self, input_label[1]) for input_label in input_labels
        ]

# create AddGeometryDialog class
class AddGeometryDialog(wx.Dialog):
    """Creates a dialog box for creating or editing geometry entries."""
    def __init__(self, parent):
        """Creates an instance of the AddGeometryDialog class."""
        super().__init__(
            parent,
            title = "Add Geometry",
            size = (1000, 300)
        )
        panel = wx.Panel(self)

        # loop over all input label-pairs
        for input_label in parent.geometry.input_labels:

            # set the relevant attribute to a newly created text box
            default = getattr(utils.Defaults, input_label[1], "")
            setattr(self, input_label[1], wx.TextCtrl(panel, value = f"{default}"))

        # layout in grid form
        grid = wx.FlexGridSizer(
            rows = len(parent.geometry.input_labels), cols = 2, hgap = 10, vgap = 8
        )
        grid.AddMany([
            item for input_label in parent.geometry.input_labels
            for item in (
                wx.StaticText(panel, label = input_label[0]),
                getattr(self, input_label[1]),
            )
        ])

        # panel-level sizer (for the form)
        panel_sizer = wx.BoxSizer(wx.VERTICAL)
        panel_sizer.Add(grid, 1, wx.ALL | wx.EXPAND, 15)
        panel.SetSizer(panel_sizer)

        # apply mainframe styling to this dialog (background + children)
        self.SetBackgroundColour(parent.colours["dark grey"])
        panel.SetBackgroundColour(parent.colours["dark grey"])
        parent.apply_styling(self)

        # dialog-level buttons
        buttons = self.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL)

        # slider row: Diffusion Factor [slider fills space] Deviation
        """slider_row = wx.BoxSizer(wx.HORIZONTAL)
        diffusion_label = wx.StaticText(panel, label = "Diffusion Factor")
        self.geometry_slider = wx.Slider(
            panel, value = int(100 * utils.Defaults.design_parameter), minValue = 114.999, maxValue = 115.001,
            style = wx.SL_HORIZONTAL
        )
        deviation_label = wx.StaticText(panel, label = "Deviation")
        slider_row.Add(diffusion_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 6)
        slider_row.Add(self.geometry_slider, 1, wx.ALL | wx.EXPAND, 6)
        slider_row.Add(deviation_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 6)
        panel_sizer.Add(slider_row, 0, wx.EXPAND)"""

        # re-apply styling so newly-created labels and slider get themed
        parent.apply_styling(self)

        # dialog root sizer
        dialog_sizer = wx.BoxSizer(wx.VERTICAL)
        dialog_sizer.Add(panel, 1, wx.EXPAND | wx.ALL, 5)
        dialog_sizer.Add(buttons, 0, wx.EXPAND | wx.ALL, 10)
        self.SetSizerAndFit(dialog_sizer)

        # store as ordered list of arguments
        self.arguments = [getattr(self, input_label[1]) for input_label in parent.geometry.input_labels]
        #self.arguments.append(self.geometry_slider)

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
