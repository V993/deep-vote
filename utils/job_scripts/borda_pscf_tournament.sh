#!/bin/bash
#SBATCH -A loni_deepvote2
#SBATCH -p gpu
#SBATCH -t 72:00:00
#SBATCH -N 1
#SBATCH -n 48
#SBATCH -o /work/lmatone/deep-voting/loni_output/efficiency_tests/borda_pscf_tournament.o
#SBATCH -e /work/lmatone/deep-voting/loni_output/efficiency_tests/borda_pscf_tournament.e
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --mail-user lmatone@tulane.edu

module load conda
cd $SLURM_SUBMIT_DIR
source activate testing

echo python3 learn_rule.py borda_pscf tournament parameters/local_job_parameters.json
python3 python3 learn_rule.py borda_pscf tournament parameters/local_job_parameters.json
echo ""

exit
