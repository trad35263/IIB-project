# import modules

import numpy as np
import matplotlib.pyplot as plt
import wx
import utils

# import custom class

from flight_scenario import Flight_scenario
from engine import Engine

# mainframe class
class MainFrame(wx.Frame):
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

        # create dropdown listing keys
        self.dropdown = wx.ComboBox(
            panel,
            choices = [str(v) for v in self.flight_scenarios.keys()],
            style = wx.CB_READONLY
        )

        # button to create new entry
        add_button = wx.Button(panel, label = "Add Flight Scenario")
        add_button.Bind(wx.EVT_BUTTON, self.add_scenario)

        # create left-hand column
        left_column = wx.BoxSizer(wx.VERTICAL)
        left_column.AddStretchSpacer()
        left_column.Add(add_button, 0, wx.ALL | wx.CENTER, 8)
        left_column.Add(self.dropdown, 0, wx.ALL | wx.EXPAND, 8)
        left_column.AddStretchSpacer()

        # create root sizer
        root = wx.BoxSizer(wx.HORIZONTAL)
        root.Add(left_column, proportion = 1, flag = wx.EXPAND | wx.ALL, border = 10)
        root.AddStretchSpacer(2)
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
                label = dialog.label.GetValue()
                arguments = [label] + [float(arg.GetValue()) for arg in dialog.arguments]
                self.flight_scenarios[label] = Flight_scenario(*arguments)

                # refresh dropdown
                self.dropdown.Append(label)

            # catch non-numeric inputs
            except ValueError:

                # display error
                wx.MessageBox("Please enter a valid numeric input.", "Invalid Input")

        # close dialog box
        dialog.Destroy()

# create ScenarioDialog class
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

        # input arguments
        self.label = wx.TextCtrl(panel)
        self.altitude = wx.TextCtrl(panel)
        self.velocity = wx.TextCtrl(panel)
        self.diameter = wx.TextCtrl(panel)
        self.hub_tip_ratio = wx.TextCtrl(panel)
        self.thrust = wx.TextCtrl(panel)

        # layout in grid form
        grid = wx.FlexGridSizer(rows=6, cols=2, hgap=10, vgap=8)
        grid.AddMany([
            (wx.StaticText(panel, label = "Label")),           (self.label),
            (wx.StaticText(panel, label = "Altitude")),        (self.altitude),
            (wx.StaticText(panel, label = "Velocity")),        (self.velocity),
            (wx.StaticText(panel, label = "Diameter")),        (self.diameter),
            (wx.StaticText(panel, label = "Hub-Tip Ratio")),   (self.hub_tip_ratio),
            (wx.StaticText(panel, label = "Thrust")),          (self.thrust),
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
            self.altitude, self.velocity,
            self.diameter, self.hub_tip_ratio, self.thrust
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
