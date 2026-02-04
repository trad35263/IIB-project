# import modules
import numpy as np
import matplotlib.pyplot as plt
import wx
import utils
import traceback

# import custom classes
from flight_scenario import Flight_scenario
from engine import Engine

# mainframe class
class MainFrame(wx.Frame):
    """Used to create the primary container for the GUI."""
    # create list of label-pairs to display on the grid
    scenario_input_labels = [
        ["Label", "label"],
        ["Altitude (m)", "altitude"],
        ["Flight Speed (m/s)", "flight_speed"],
        ["Diameter (m)", "diameter"],
        ["Hub-tip Ratio", "hub_tip_ratio"],
        ["Thrust (N)", "thrust"]
    ]
    scenario_display_labels = (
        scenario_input_labels + [
            ["Mach Number", "M"],
            ["Thrust Coefficient", "C_th"]
        ]
    )
    engine_input_labels = [
        ["No. of Stages", "no_of_stages"],
        ["Vortex Exponent", "vortex_exponent"],
        ["Solver Order", "solver_order"],
        ["Stagnation Pressure Loss Coefficient", "Y_p"],
        ["Flow Coefficient", "phi"],
        ["Stage Loading Coefficient", "psi"]
    ]
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
    def __init__(self):
        """Creates instance of the MainFrame class."""
        # create gui and specify title and size
        super().__init__(
            parent = None,
            title = "High Speed Solver",
            size = (1000, 600)
        )

        # store Flight_scenario instances as dictionary
        self.flight_scenarios = {}

        # loop over key-value pairs of default input arguments
        for key, values in utils.Defaults.flight_scenarios.items():

            # take default arguments and construct corresponding class
            self.flight_scenarios[key] = Flight_scenario(*values)

        # create panel
        panel = wx.Panel(self)

        # create dropdown listing flight scenarios and bind to event function
        self.scenario_dropdown = wx.ComboBox(
            panel,
            choices = [str(key) for key in self.flight_scenarios.keys()],
            style = wx.CB_READONLY,
            value = str(next(iter(self.flight_scenarios)))
        )
        self.scenario_dropdown.Bind(wx.EVT_COMBOBOX, self.change_scenario)

        # create dropdown listing associated engines and bind to event function
        self.engine_dropdown = wx.ComboBox(
            panel,
            choices = [],
            style = wx.CB_READONLY
        )
        self.engine_dropdown.Bind(wx.EVT_COMBOBOX, self.change_engine)

        # create button to add new flight scenarios and bind to event function
        self.add_scenario_button = wx.Button(panel, label = "Add Flight Scenario")
        self.add_scenario_button.Bind(wx.EVT_BUTTON, self.add_scenario)

        # create button to add new engines and bind to event function
        self.add_engine_button = wx.Button(panel, label = "Create Engine")
        self.add_engine_button.Bind(wx.EVT_BUTTON, self.add_engine)

        # create buttons for each Engine class method
        methods = [name for name, f in Engine.__dict__.items() if callable(f) and "plot" in name]
        buttons = [wx.Button(panel, label = method) for method in methods]
        for button in buttons:

            # bind button to event function
            button.Bind(wx.EVT_BUTTON, self.display_plot)

        # create empty lists of text boxes with values to be edited
        self.scenario_display_texts = []
        self.engine_display_texts = []

        # create a 2-column grid for flight scenario details
        self.scenario_grid = wx.FlexGridSizer(
            rows = len(self.scenario_display_labels), cols = 2, hgap = 10, vgap = 5
        )
        self.scenario_label = self.scenario_dropdown.GetValue()

        # create a 2-column grid for engine details
        self.engine_grid = wx.FlexGridSizer(
            rows = len(self.engine_display_labels), cols = 2, hgap = 10, vgap = 5
        )

        # loop over label-pairs for scenarios
        for (display, attribute) in self.scenario_display_labels:

            # add text and store value
            self.scenario_grid.Add(wx.StaticText(panel, label = display))
            value = getattr(self.flight_scenarios[self.scenario_label], attribute)

            # for string attributes
            if isinstance(value, str):

                # display as is
                text = wx.StaticText(panel, label = f"{value}")

            # for numeric attributes
            else:

                # display to 4 significant figures
                text = wx.StaticText(panel, label = f"{value:.4g}")

            # store text object and add to grid
            self.scenario_display_texts.append(text)
            self.scenario_grid.Add(text)

        # loop over label-pairs for engines
        for (display, attribute) in self.engine_display_labels:

            # add text and store value
            self.engine_grid.Add(wx.StaticText(panel, label = display))
            text = wx.StaticText(panel, label = "")

            # store text object and add to grid
            self.engine_display_texts.append(text)
            self.engine_grid.Add(text)

        # create left-hand column
        left_column = wx.BoxSizer(wx.VERTICAL)

        # assign button, dropdown and grid to left-hand column
        left_column.Add(self.add_scenario_button, 0, wx.ALL | wx.CENTER, 8)
        left_column.Add(self.scenario_dropdown, 0, wx.ALL | wx.EXPAND, 8)
        left_column.Add(self.scenario_grid, 0, wx.ALL | wx.CENTER, 8)
        left_column.AddStretchSpacer()

        # create central column
        centre_column = wx.BoxSizer(wx.VERTICAL)

        # assign button, dropdown and grid to centre column
        centre_column.Add(self.add_engine_button, 0, wx.ALL | wx.CENTER, 8)
        centre_column.Add(self.engine_dropdown, 0, wx.ALL | wx.EXPAND, 8)
        centre_column.Add(self.engine_grid, 0, wx.ALL | wx.CENTER, 8)
        centre_column.AddStretchSpacer()

        # create right-hand column
        right_column = wx.BoxSizer(wx.VERTICAL)

        # assign buttons to right-hand column
        for button in buttons:

            right_column.Add(button, 0, wx.ALL | wx.EXPAND, 8)

        # create root sizer
        root = wx.BoxSizer(wx.HORIZONTAL)
        root.Add(left_column, proportion = 1, flag = wx.EXPAND | wx.ALL, border = 10)
        root.Add(centre_column, proportion = 1, flag = wx.EXPAND | wx.ALL, border = 10)
        root.Add(right_column, proportion = 1, flag = wx.EXPAND | wx.ALL, border = 10)
        panel.SetSizer(root)

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
                f"N: {engine.no_of_annuli}"
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

        # no engines exist
        else:

            # loop over all engine information texts
            for text in self.engine_display_texts:

                # clear labels
                text.SetLabel("")

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
                self.flight_scenarios[label].engines.append(Engine(*arguments))

                # refresh dropdown
                no_of_stages = int(dialog.arguments[0].GetValue())
                vortex_exponent = float(dialog.arguments[1].GetValue())
                no_of_annuli = int(dialog.arguments[2].GetValue())
                self.engine_dropdown.Append(
                    f"[{len(self.flight_scenarios[label].engines) - 1}]"
                    f"      Stages: {no_of_stages} | n: {vortex_exponent} | N: {no_of_annuli}"
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

        # get latest label from engine dropdown
        self.engine_label = self.engine_dropdown.GetValue()
        self.engine_label = int(self.engine_label.split(']')[0][1:])

        # loop over pairs of scenario labels and text boxes
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

# create AddScenarioDialog class
class AddScenarioDialog(wx.Dialog):
    """Used to create a dialog box for inputting arguments for additional Flight_scenario classes."""
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
    """Used to create a dialog box for creating an additional Engine class."""
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
