# CFD_validation.py
# 24 May 2026

# import modules
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import ast

# import high speed solver
import utils
from engine import Engine
from flight_scenario import Flight_scenario

import importlib.util

# 1. Add the parent directory to your system path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 2. Point directly to the file using the hyphenated folder name
#    (Adjust 'run_analysis.py''s relative depth if necessary)
module_path = os.path.join(parent_dir, "trad3-multistage-fans", "smith_post_process.py")

# 3. Programmatically load the module
spec = importlib.util.spec_from_file_location("smith_post_process", module_path)
smith_post_process = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smith_post_process)

# 4. Extract 'Post' from the dynamically loaded module
Post = smith_post_process.Post

# Inputs class
class Inputs:

	# file path to TURBOSTREAM folder containing post-processed packages of data
	folder_path = "C:/Users/tomra/OneDrive/Documents/Uni/2B Coursework/IIB Project/CFD Data/"

	# default phi-psi coordinates
	phi_psi = None

	# 
	gamma = 1.4
	R = 287
	

# main function
def main():
	
    # 
	smith_post_process.Inputs = Inputs
	post = Post()

# upon script execution
if __name__ == "__main__":

	# no additional input arguments are provided
	if len(sys.argv) < 2:

		# print error message
		print(f"{utils.Colours.RED}Please provide a data folder location!{utils.Colours.END}")

	# input folder is specified
	else:

		# store input folder
		Inputs.folder = os.path.join(Inputs.folder_path, sys.argv[1])
		print(f"Reading from folder: {utils.Colours.GREEN}{Inputs.folder}{utils.Colours.END}")

	# a specific test case is provided
	if len(sys.argv) > 2:

		# store phi-psi coordinates of test case
		Inputs.phi_psi = ast.literal_eval(sys.argv[2])
		print(f"Analysing test case with phi-psi coordinates: {utils.Colours.GREEN}{Inputs.phi_psi}{utils.Colours.END}")

	# run main()
	main()
	plt.show()
