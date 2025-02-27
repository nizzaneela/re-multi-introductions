#!/usr/bin/env python3

"""
Replicates Pekar et al.'s single introduction simulations and analysis deterministically.
Simulations differ from Pekar et al.'s in that the primary case starts in the exposed state.
Analysis differs from Pekar et al.'s in that the stable coalescent is not used.
Implementation differs from Pekar et al.'s to reduce memory demands and computation cost.
Results differ from Pekar et al.'s mainly because of reduced sampling error.
"""

import numpy as np # 1.19 or later
import concurrent.futures
import sys
import os
import shutil

from one_intro_functions import *
from two_intro_functions import *

def main(parameters_file_name: str, num_processors: int):
    """
    Runs one_intro simulations and analyzes them.
    Samples pairs of one_intro simulations as two_intro simulations and analyzes them.

    :param parameters_file_name: GEMF parameter file name.
    :param num_processors: Number of parallel processes.
    """
    # === General setup ===
    # rng seeds
    seed = 42
    one_intro_seedseq, two_intro_seedseq = np.random.SeedSequence(seed).spawn(2)
    # ensure analysis directory is cwd
    analysis_dir  = f"{parameters_file_name.split('_')[0]}_analysis"
    if os.path.basename(os.getcwd()) != analysis_dir:
        os.makedirs(analysis_dir, exist_ok=True)
        shutil.copy2(parameters_file_name, os.path.join(analysis_dir, parameters_file_name))
        os.chdir(analysis_dir)

    # === Single introduction simulation and analysis ===
    # prepare lists to map to worker arguments
    directory_list = []
    parameter_list = []
    for i in range(NUM_SIMS):
        directory_list.append(os.path.join("one_intro", str(i)))
        parameter_list.append(parameters_file_name)
    seedseq_list = one_intro_seedseq.spawn(NUM_SIMS)
    # run workers - one for each successful introduction and its analysis
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processors) as executor:
        futures = []
        for directory, param, seedseq in zip(directory_list, parameter_list, seedseq_list):
            futures.append(executor.submit(one_intro, directory, param, seedseq))
        concurrent.futures.wait(futures)

    # === Two introduction sampling and analysis ===
    # prepare lists to map to worker arguments
    directory_list = []
    parameter_list = []
    rng_list = []
    intro_0_list = []
    intro_1_list = []
    for i, seq in enumerate(two_intro_seedseq.spawn(NUM_SIMS)):
        directory_list.append(os.path.join("two_intro", str(i)))
        parameter_list.append(parameters_file_name)
        rng = np.random.default_rng(seq)
        id_0 = rng.integers(NUM_SIMS)
        id_1 = rng.integers(NUM_SIMS)
        rng_list.append(rng)
        intro_0_list.append(id_0)
        intro_1_list.append(id_1)
    # run workers - one for each pair of successful introductions and its analysis
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processors) as executor:
        futures = []
        for directory, param, rng, id_0, id_1 in zip(directory_list, parameter_list, rng_list, intro_0_list, intro_1_list):
            futures.append(executor.submit(two_intro, directory, param, rng, id_0, id_1))
        concurrent.futures.wait(futures)



if __name__ == "__main__":
    # General form
    # # We accept 2 arguments total:
    # #   1) <parameters_file_name>
    # #   2) <num_processors>
    # if len(sys.argv) != 3:
    #     print("Usage: python3 script.py <parameters_file_name> <num_processors>")
    #     sys.exit(1)
    #
    # parameters_file_name = 'sys.argv[1]'
    # num_processors = int(sys.argv[2])

    # main specific form
    # We accept 1 arguments total:
    #   1) <num_processors>
    if len(sys.argv) != 2:
        print("Usage: python3 script.py <num_processors>")
        sys.exit(1)

    parameters_file_name = 'main_parameters.txt'
    num_processors = int(sys.argv[1])

    main(parameters_file_name, num_processors)
