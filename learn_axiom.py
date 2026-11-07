
from data import load_pregenerated_data
from learn_rule import training_dispatcher

from utils import (
    load_job_parameters,
    save_experiment
)

device = "cpu"

rule_name = "borda_pscf"
transform_name = "concatenate"
params = load_job_parameters(rule_name, transform_name, "utils/parameters/11c44v_120kp_learn_participation_transfer.json")

params["n_profiles"] = 1048
params["batch_size"] = 1
params["loss_fn"] = "MajorityWinnerLoss"
params["n_epochs"] = 10

# Load specific distributions for majority winner testing:
# params["data_filename"] = "data/pregenerated/7c_29v_512p_mallows.h5"
# params["models_directory"] = "experiments/7c_29v_120000p_L1Loss_concatenate_large/models"

train_generator, _, _, = load_pregenerated_data(
    params, proportions=(1.0, 0.0)
)

first = next(iter(train_generator))
first

first[0].get("profile").shape
# torch.Size([1, 44, 11]), This is a single batch with 44 voters nad 11 candidates

model, train_losses, valid_losses, training_batch_losses, valid_batch_losses = training_dispatcher(
    rule_name, 
    transform_name, 
    train_generator, 
    valid_generator=None, 
    device=device, 
    params=params
)

save_experiment(
    model, 
    train_losses, 
    params, 
    train_batches=training_batch_losses,
)
