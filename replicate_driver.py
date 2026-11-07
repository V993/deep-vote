""" REPLICATE JOB DRIVER

Submits every (rule, transform) job N times so that per-cell variance can be estimated.

Each replicate is passed to learn_rule.py as a 4th argument, which lands in params["replicate"] and makes
get_experiment_directory() write to '<experiment_dir>_rep<n>'. Without that suffix all replicates resolve
to the same directory and silently overwrite each other's models, losses and metadata.

The transform set follows the config: a concatenate config trains one model per rule, otherwise one model
per rule per single transform. Concatenate runs deliberately share an experiment directory with the
single-transform runs -- job names are '<rule>_<transform>' so the filenames do not collide.

Nothing is submitted until every job's SLURM name, script filename and log path are confirmed distinct.

    Usage:
        python replicate_driver.py <parameter_filename> [--replicates N] [--dry-run] [--local]

    Typical Table 3 sweep (160 jobs total):
        python replicate_driver.py utils/parameters/11c44v_120kp_learn_rules.json --replicates 5
        python replicate_driver.py utils/parameters/11c44v_120kp_learn_rules_concatenate.json --replicates 5
"""

import argparse
import itertools
import json
import os

RULES = [
    "black_pscf",
    "borda_pscf",
    "copeland_pscf",
    "irv_pscf",
    "maximin_pscf",
    "plurality_pscf",
    "rankedpairs_pscf",
    "schulze_pscf",
]

SINGLE_TRANSFORMS = ["tournament", "weighted_tournament", "rankfreq"]

# Kept out of utils/job_scripts/ directly so these never overwrite scripts written by rules_driver.py
SCRIPT_DIR = "utils/job_scripts/replicates"


def build_loni_script(loni_details):
    with open("utils/job_scripts/job_template.txt", 'r') as f:
        text = "".join(f.readlines())
        text = text.format(**loni_details)
    return text


def load_parameters(params_file):
    with open(params_file, 'r') as f:
        return json.load(f)


def job_identifiers(parameter_file, parameters, rule, transform, replicate):
    """Every name this job writes to. Checked for uniqueness before anything is submitted."""
    param_base = os.path.splitext(os.path.basename(parameter_file))[0]
    experiment_type = parameters.get("experiment_type")
    stem = f"{rule}_{transform}_rep{replicate}"
    return {
        "job_name": f"{experiment_type}-{rule}-{transform}-rep{replicate}",
        "outfile": f"{experiment_type}/{param_base}_{stem}",
        "script_filename": f"{SCRIPT_DIR}/{stem}.sh",
    }


def transforms_for(parameters):
    """A concatenate config trains a single concatenated model per rule; otherwise one per transform."""
    if bool(parameters.get("concatenate")) is True:
        return ["concatenate"]
    return list(SINGLE_TRANSFORMS)


def build_jobs(rules, transforms, replicates, start=1):
    return [
        (rule, transform, replicate)
        for replicate in range(start, replicates + 1)
        for rule, transform in itertools.product(rules, transforms)
    ]


def find_collisions(jobs, parameter_file, parameters):
    """Return a list of human-readable collision descriptions; empty means every name is distinct."""
    seen = {}
    collisions = []
    for rule, transform, replicate in jobs:
        names = job_identifiers(parameter_file, parameters, rule, transform, replicate)
        for kind, value in names.items():
            key = (kind, value)
            if key in seen:
                collisions.append(f"{kind} '{value}' claimed by both {seen[key]} and {(rule, transform, replicate)}")
            seen[key] = (rule, transform, replicate)
    return collisions, len(seen)


def model_path(parameters, rule, transform, replicate):
    """Where learn_rule.py will save this job's checkpoint (mirrors get_experiment_directory without creating dirs)."""
    p = dict(parameters)
    p.setdefault("pref_dist", "impartial_culture")  # learn_rule.py applies the same default
    exp_dir = "experiments/{n_cands}c_{n_voters}v_{n_profiles}p_{pref_dist}_{loss_fn}_{experiment_type}".format(**p)
    return f"{exp_dir}_rep{replicate}/models/{rule}_{transform}.pt"


def existing_outputs(jobs, parameters):
    """Checkpoints that a submission would overwrite."""
    return [path for rule, transform, replicate in jobs
            if os.path.exists(path := model_path(parameters, rule, transform, replicate))]


def submit_loni(i, rule, transform, replicate, parameter_file, parameters):
    print(f"{i}: Submitting ({rule}, {transform}) rep {replicate} on {parameters.get('partition')}")

    names = job_identifiers(parameter_file, parameters, rule, transform, replicate)
    parameters["outfile"] = names["outfile"]
    parameters["command"] = f"python3 learn_rule.py {rule} {transform} {parameter_file} {replicate}"
    parameters["job_name"] = names["job_name"]

    with open(names["script_filename"], 'w') as f:
        f.write(build_loni_script(parameters))

    os.system(f"sbatch {names['script_filename']}")


def submit_local(i, rule, transform, replicate, parameter_file):
    print(f"{i}: Running ({rule}, {transform}) rep {replicate} LOCALLY")
    os.system(f"python3 learn_rule.py {rule} {transform} {parameter_file} {replicate}")


def parse_args():
    parser = argparse.ArgumentParser(description="Submit every rule/transform job N times for variance estimation.")
    parser.add_argument("parameter_file", help="Path to the job parameter JSON file.")
    parser.add_argument("--replicates", type=int, default=5, help="How many times to run each job (default 5).")
    parser.add_argument("--start-replicate", type=int, default=1,
                        help="First replicate id to submit (default 1); e.g. --start-replicate 2 --replicates 5 submits reps 2-5.")
    parser.add_argument("--dry-run", action="store_true", help="Print every job that would be submitted, then exit.")
    parser.add_argument("--local", action="store_true", help="Run jobs locally instead of submitting to SLURM.")
    parser.add_argument("--rules", nargs="+", default=None, help="Restrict to these rules (default: all 8).")
    parser.add_argument("--allow-existing", action="store_true",
                        help="Submit even if a job's model checkpoint already exists (e.g. to rerun failed jobs).")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.replicates < 1 or not 1 <= args.start_replicate <= args.replicates:
        print("--replicates must be at least 1, and --start-replicate between 1 and --replicates")
        raise SystemExit(1)

    params = load_parameters(args.parameter_file)
    rules = args.rules if args.rules else list(RULES)
    transforms = transforms_for(params)
    jobs = build_jobs(rules, transforms, args.replicates, args.start_replicate)

    collisions, n_names = find_collisions(jobs, args.parameter_file, params)
    if collisions:
        print(f"Refusing to submit: {len(collisions)} name collision(s) detected.")
        for c in collisions:
            print(f"    {c}")
        raise SystemExit(1)

    existing = existing_outputs(jobs, params)
    if existing and not args.allow_existing:
        print(f"Refusing to submit: {len(existing)} of {len(jobs)} jobs would overwrite an existing checkpoint, e.g.")
        for path in existing[:5]:
            print(f"    {path}")
        print("Move the previous run aside first (bash utils/job_scripts/archive_replicates.sh <tag> --apply),")
        print("or pass --allow-existing to overwrite on purpose (e.g. rerunning failed jobs).")
        raise SystemExit(1)

    print(f"{len(rules)} rules x {len(transforms)} transform(s) x {args.replicates - args.start_replicate + 1} replicate(s) = {len(jobs)} jobs")
    print(f"Experiment dirs will be suffixed _rep{args.start_replicate}.._rep{args.replicates}")
    print(f"All {n_names} job/script/log names verified distinct.\n")

    if args.dry_run:
        for i, (rule, transform, replicate) in enumerate(jobs):
            names = job_identifiers(args.parameter_file, params, rule, transform, replicate)
            print(f"{i}: {names['job_name']}")
            print(f"     script : {names['script_filename']}")
            print(f"     logs   : loni_output/{names['outfile']}.o/.e")
            print(f"     command: python3 learn_rule.py {rule} {transform} {args.parameter_file} {replicate}")
        print(f"\nDry run: {len(jobs)} jobs NOT submitted.")
        return

    os.makedirs(SCRIPT_DIR, exist_ok=True)
    # sbatch fails if the log directory does not exist, which is easy to miss across 160 jobs
    os.makedirs(f"loni_output/{params.get('experiment_type')}", exist_ok=True)

    for i, (rule, transform, replicate) in enumerate(jobs):
        if args.local:
            submit_local(i, rule, transform, replicate, args.parameter_file)
        else:
            submit_loni(i, rule, transform, replicate, args.parameter_file, params)


if __name__ == "__main__":
    main()
