#!/usr/bin/env python3

"""
Replicates Pekar et al.'s single introduction simulations and analysis deterministically.
Simulations differ from Pekar et al.'s in that the primary case starts in the exposed state.
Analysis differs from Pekar et al.'s in that the stable coalescent is not used.
Implementation differs from Pekar et al.'s to reduce memory demands and computation cost.
Results differ from Pekar et al.'s mainly because of reduced sampling error.
"""

import numpy as np # 1.19 or later
import os
import subprocess
import treeswift

from shared_functions import *

def simulate_epidemic(
    output_dir: str,
    parameters_file_name: str,
    rng: np.random.Generator
):
    """
    Copy the GEMF input parameters file from current into simulation directory,
    inserting a random integer seed in place of the [RANDOM_SEED] placeholder,
    then run GEMF.

    :param output_dir: Directory for simulation outputs.
    :param parameters_file_name: Name of the GEMF parameter file.
    :param rng: Numpy random generator.
    :return rng: The same RNG for chaining.
    """
    # 1. Modify parameter file with a new random seed
    with open(parameters_file_name, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip() == "[RANDOM_SEED]":
            lines[i + 1] = str(rng.integers(2**32)) + "\n"
            break

    # 2. Write out updated parameter file in the simulation directory
    updated_params_path = os.path.join(output_dir, parameters_file_name)
    with open(updated_params_path, 'w') as f:
        f.writelines(lines)

    # 3. Run GEMF
    command = f"GEMF {parameters_file_name}"
    logfile_path = os.path.join(output_dir, "GEMF_log.txt")
    with open(logfile_path, 'w') as logfile:
        subprocess.run(command, cwd=output_dir, shell=True, stdout=logfile, stderr=logfile)

    return rng


def one_intro(output_dir, parameters_file_name, seedseq):
    """
    Run a single-introduction simulation and analyze it.

    :param output_dir: Directory for outputs.
    :param parameters_file_name: The GEMF parameters file.
    :param seedseq: Seed sequence for numpy random generator.
    """
    rng = np.random.default_rng(seedseq)
    os.makedirs(output_dir, exist_ok=True)
    sim_end_time = get_sim_end_time(parameters_file_name)

    # 1. Run GEMF to generate transmission network
    rng = simulate_epidemic(output_dir, parameters_file_name, rng)

    # 2. Generate sample times
    rng = simulate_sequencing(output_dir, sim_end_time, rng)

    # 3. Generate viral phylogeny
    rng = simulate_coalescence(output_dir, rng)

    # 4. Simulate mutations and analyze clades
    AB, CC, tree = analyze_clades(output_dir, rng)

    # Write AB result to file
    with open(os.path.join(output_dir, 'AB.txt'), 'w') as f:
        f.write(f"{AB}\n")

    # Write CC result to file
    with open(os.path.join(output_dir, 'CC.txt'), 'w') as f:
        f.write(f"{CC}\n")

    # store the final tree
    tree.write_tree_newick(os.path.join(output_dir, "final_tree.nwk"))

    # store a hash of the final tree, for verification
    output_hash = compute_tree_hash(tree.root)
    with open(os.path.join(output_dir, 'final_tree_hash.txt'), 'w') as f:
        f.write(output_hash)

