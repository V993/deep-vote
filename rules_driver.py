

""" RULE LEARNING JOB DRIVER
    Usage: python rules_driver.py <parameter_filename> [loni_bool]

    Arguments:
        <parameter_filename>  Path to the parameter file for the job.
        [loni_bool] (optional)  Whether or not to run scripts locally or on LONI. True or not-passed for LONI

    TODO: should utilize more GPUs atomically instead of reserving an entire GPU node and using only 1 GPU
    TODO: "full" is not currently supported. We have neither a network nor the profile to process a full profile currently.
"""

import os
import json
import itertools
from sys import argv


def build_loni_script(loni_details):
    with open("utils/job_scripts/job_template.txt", 'r') as f:
        text = "".join(f.readlines())
        text = text.format(**loni_details)
    return text


def usage():
    print(f"Usage: python {argv[0]} <parameter_filename> [loni_bool]\n")
    print("Arguments:")
    print("    <parameter_filename>  Path to the parameter file for the job.")
    print("    [loni_bool] (optional)  Whether or not to run scripts locally or on LONI. Defaults to LONI")


def handle_local(i, rule, transform, parameter_file):
    print(f"{i}: Submitting ({rule}, {transform}) on LOCAL")
    os.system(f"python3 learn_rule.py {rule} {transform} {parameter_file}")


def handle_loni(i, rule, transform, parameter_file, parameters):
    print(f"{i}: Submitting ({rule}, {transform}) on {parameters.get('partition')}")
    
    script = "learn_rule.py"  # Default: learn rules/axioms
    if parameters.get("use_whalrus_rule_module", False):
        script = "evaluate_whalrus_rule.py"

    param_base = os.path.splitext(os.path.basename(parameter_file))[0]
    parameters["outfile"] = f"{parameters.get('experiment_type')}/{param_base}_{rule}_{transform}"
    parameters["command"] = f"python3 {script} {rule} {transform} {parameter_file}"
    parameters["job_name"] = f"{parameters.get('experiment_type')}-{rule}-{transform}"
    
    script_content = build_loni_script(parameters)
    script_filename = f"utils/job_scripts/{rule}_{transform}.sh"
    with open(script_filename, 'w') as f: f.write(script_content)

    os.system(f"sbatch {script_filename}")


def load_parameters(params_file, keys=None):
    with open(params_file, 'r') as f:
        params = json.load(f)

    if keys==None:
        keys = list(params.keys())
    
    return {key: params[key] for key in keys}


def main():

    # Check for parameters and partition
    if len(argv) < 2:
        usage()
        # parameters = "parameters/local_job_parameters.json"
        exit(1)
    
    parameter_file = argv[1]

    loni = True
    if len(argv) > 2:
        loni = False if argv[2].lower().strip() == "false" else True

    # Load parameter file
    params = load_parameters(parameter_file)

    # Define the rules/transforms for the experiment:
    # rules = [ "borda_pscf" ]
    transforms = [ "concatenate" ]
    # rules = ["black_pscf", "maximin_pscf", "borda_pscf", "copeland_pscf", "irv_pscf", "plurality_pscf", "rankedpairs_pscf", "schulze_pscf"]
    rules = ["none"]

    # if bool(params.get("concatenate")) == True:
    #     transforms = ["concatenate"]
    # else:
    #     transforms = ["tournament", "weighted_tournament", "rankfreq"] # "full"
    

    # Get all combinations as the jobs to be submitted:
    jobs = list(itertools.product(rules, transforms))

    # Create and submit jobs
    for i, (rule, transform) in enumerate(jobs):
        if loni:
            handle_loni(i, rule, transform, parameter_file, params)
        else:
            handle_local(i, rule, transform, parameter_file)

if __name__ == "__main__":
    main()
