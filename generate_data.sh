#!/bin/bash
#SBATCH -A loni_ethicllms
#SBATCH -p workq
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -n 48
#SBATCH -o /work/lmatone/deep-voting/loni_output/generated_data_abstensions.o
#SBATCH -e /work/lmatone/deep-voting/loni_output/generated_data_abstensions.e
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --mail-user lmatone@tulane.edu

module load conda
cd $SLURM_SUBMIT_DIR

# Go into the foundation model directory for our new data generation routine
cd foundation_model

source activate testing

# If we have a param from caller, pass it. Otherwise default to 29c99v_120kp_learn_rules.json
if [ -z "$1" ]; then
    echo "No parameter file provided, using default: 29c99v_120kp_learn_rules.json"
    PARAM_FILE="utils/parameters/29c99v_120kp_learn_rules.json"
else
    echo "Using provided parameter file: $1"
    PARAM_FILE=$1
fi

# python3 generate_data.py utils/parameters/29c99v_120kp_learn_rules.json
python3 generate_data.py
echo ""

exit
