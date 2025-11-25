# xml_exporter.py

# import modules
from datetime import datetime, timezone
import numpy as np
import os
import sys
from scipy.spatial import distance

# create class for storing colours
class Colours:
    """Class used to store ANSI escape sequences for printing colours."""
    # store ASCII codes for selected colours as class attributes
    RED = '\033[91m'
    ORANGE = '\033[38;5;208m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    PINK = '\033[38;5;212m'
    GREY = '\033[90m'
    BOLD = '\033[1m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'
    STRIKETHROUGH = '\033[9m'
    END = '\033[0m'

# xml_exporter function
def xml_exporter(data, label = False):
    """Exports an input of x-y data pairs in XML format for import to draw.io."""
    # store colours class in a convenient variable
    c = Colours()

    # adjust data
    """D = distance.pdist(data)
    min_index = np.argmin(D)
    i, j = np.triu_indices(len(data), 1)
    closest_pair = (i[min_index], j[min_index])
    data -= data[closest_pair[0]]"""
    #data *= 320 / (np.max(data[:, 0]) - np.min(data[:, 0]))

    # store current data and time and determine a name for the file
    date_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    name = "xml_" + date_time[0:10] + (f"-{label}" if label else "")

    # create xml string with necessary preamble
    xml = f"""
<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="app.diagrams.net" modified="{date_time}" version="20.6.0" type="device">
    <diagram id="xml_exporter_export" name="{name}">
        <mxGraphModel>
            <root>
                <mxCell id="0"/>
                <mxCell id="1" parent="0"/>
                <mxCell id="2" style="edgeStyle=none;endArrow=none;strokeColor=#0000ff;strokeWidth=2;" edge="1" parent="1">
                    <mxGeometry relative="1" as="geometry">
                        <mxPoint x="{data[0][0]}" y="{data[0][1]}" as="sourcePoint"/>
                        <mxPoint x="{data[0][0]}" y="{data[0][1]}" as="targetPoint"/>
                        <Array as="points">
"""
    
    # iterate over all pairs of x-y data
    for [x, y] in data:

        # append x-y datapoint to xml file
        xml += f"""
                            <mxPoint x="{x}" y="{-y}"/>"""

    # close all necessary brackets in xml file
    xml += f"""

                        </Array>
                    </mxGeometry>
                </mxCell>
            </root>
        </mxGraphModel>
    </diagram>
</mxfile>
"""
    
    # define output sub-folder and create folder if it doesn't exist
    output_dir = os.path.join(os.getcwd(), "XML_exports")
    os.makedirs(output_dir, exist_ok = True)

    # define output file path
    filename = os.path.join(
        output_dir,
        os.path.splitext(os.path.basename(name))[0] + ".xml"
    )

    # create text file
    with open(filename, "w", encoding = "utf-8") as file:

        # write xml output to file
        file.write(xml)
    
    # print feedback to user
    print(f"{c.GREEN}XML exported to {filename}.{c.END}")

def main():
    # store colours class in a convenient variable
    c = Colours()

    # check if insufficient arguments are provided to the terminal
    if len(sys.argv) < 2:

        # print the expected syntax to the user and terminate the script
        print(
            f"{c.RED}Please use the following syntax:{c.END} \n"
            f"python xml_exporter.py <{c.CYAN}data_file.npz{c.END}> [label]"
        )
        sys.exit(1)

    # store filename and label from provided arguments
    filename = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else filename.replace(".npz", "")

    # try-except block to handle errors
    try:

        # load .npz file
        datafile = np.load(filename)

        # check if x and y are provided as individual arrays
        if "x" in datafile and "y" in datafile:

            # extract x-y pairs and store in shape (N, 2)
            data = np.column_stack((datafile["x"], datafile["y"]))

        #  check if x and y are provided in a single array of shape (N, 2)
        else:

            # extract x-y pairs and store in shape (N, 2
            data = list(datafile.values())[0]
    
    # handle errors
    except Exception as error:

        # print filename and error and terminate script
        print(f"{c.RED}Error loading {filename}:{c.END} \n{error}")
        sys.exit(1)

    # run xml_exporter() function
    xml_exporter(data, label)

# on script execution
if __name__ == "__main__":

    # uncomment this to convert .csv to .npz
    """data = []
    with open("naca644421-il.csv") as f:
        for line in f:
            try:
                numbers = [float(x) for x in line.strip().split(",")]
                data.append(numbers)
            except ValueError:
                # Skip lines that contain non-numeric data
                continue

    data = np.array(data)
    print(data)
    print(data.shape)
    np.savez("turbine_stator.npz", data=data)
    xml_exporter(data, "turbine_stator")
    input()"""

    # run main() on script execution
    main()