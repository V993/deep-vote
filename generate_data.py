
import json
from sys import argv
from data import generate_data_wrapper

def main():

    parameter_file = "utils/parameters/local_job_parameters.json"
    if len(argv) >= 2: 
        parameter_file = argv[1]
        print(f"Loading parameters from '{parameter_file}'")
    
    # Generate and save data to h5 dataset:
    generated_data = generate_data_wrapper(parameter_file)

if __name__ == "__main__":
    main()
