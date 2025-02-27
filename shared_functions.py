#!/usr/bin/env python3

"""
Replicates Pekar et al.'s single introduction simulations and analysis deterministically.
Simulations differ from Pekar et al.'s in that the primary case starts in the exposed state.
Analysis differs from Pekar et al.'s in that the stable coalescent is not used.
Implementation differs from Pekar et al.'s to reduce memory demands and computation cost.
Results differ from Pekar et al.'s mainly because of reduced sampling error.
"""

import numpy as np # 1.19 or later
import sys
import os
import shutil
import subprocess
import treeswift
import hashlib
import copy

N_SITES = 29903 # substitution sites
CLOCK = 0.00092 # clock rate

NUM_SIMS = 110000 # 100x Pekar's sample size


def get_sim_end_time(parameter_file_path: str):
    # Get sim_end_time from the GEMF input parameters file
    with open(parameter_file_path, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip() == "[MAX_TIME]":
            sim_end_time = float(lines[i + 1].strip())
            break
    return sim_end_time


def simulate_sequencing(
    output_dir: str,
    sim_end_time: float,
    rng: np.random.Generator
):
    """
    Read epidemic simulation output and create sample_times.txt (a list of times when the virus samples were taken from ascertained cases and sequenced). The sampling time is uniformly drawn
    between start (ascertainment) and end (recovery or sim_end_time).
    Also ensures that any hospitalised case which was ascertained before the first hospitalization
    is only sampled after it.

    :param output_dir: Directory containing simulation outputs (disease_events.txt, etc.).
    :param parameters_file_name: GEMF parameter file name.
    :param rng: Numpy random generator.
    :return: The same RNG (for chaining).
    """
    ascertained_cases = {}
    first_hospitalisation_time = 0.0

    # Read disease info from disease_events.txt
    with open(os.path.join(output_dir, 'disease_events.txt'), 'r') as f:
        for line in f:
            parts = line.strip().split()
            event_time = float(parts[0])
            infection_id = parts[1]
            event_type = parts[3]

            if event_type == '3':  # ascertainment event
                ascertained_cases[infection_id] = {'start_time': event_time, 'end_time': 0.0}
            elif event_type == '7':  # recovery event
                if infection_id in ascertained_cases:
                    ascertained_cases[infection_id]['end_time'] = event_time
            elif event_type == '6':  # hospitalisation event
                # record first hospitalisation time
                if first_hospitalisation_time == 0:
                    first_hospitalisation_time = event_time
                # ensure start time does not precede first hospitalisation time
                if ascertained_cases[infection_id]['start_time'] < first_hospitalisation_time:
                    ascertained_cases[infection_id]['start_time'] = first_hospitalisation_time

    # Generate and write out sample times
    sample_times_path = os.path.join(output_dir, 'sample_times.txt')
    with open(sample_times_path, 'w') as f:
        for infection_id, timing in ascertained_cases.items():
            start_time = timing['start_time']
            end_time = timing['end_time'] if timing['end_time'] != 0 else sim_end_time
            sample_time = rng.uniform(start_time, end_time)

            # Only write sample time if it is after first hospitalisation
            if sample_time > first_hospitalisation_time:
                f.write(f"{infection_id}\t{sample_time}\n")

    return rng


def simulate_coalescence(output_dir: str, rng: np.random.Generator):
    """
    Run CoaTran to generate the time-scaled tree.

    :param output_dir: Directory containing transmission_network.txt, sample_times.txt, etc.
    :param rng: Numpy random generator.
    :return: The same RNG (for chaining).
    """
    coatran_rng_seed = rng.integers(2**32)
    command = "coatran_constant transmission_network.txt sample_times.txt 1"
    env = os.environ.copy()
    env["COATRAN_RNG_SEED"] = str(coatran_rng_seed)
    tree_path = os.path.join(output_dir, "tree.time.nwk")
    with open(tree_path, 'w') as output_file:
        subprocess.run(
            command,
            cwd=output_dir,
            shell=True,
            stdout=output_file,
            env=env
        )

    return rng


def get_clade_info(parent: treeswift.Node, rng: np.random.Generator):
    """
    Recursively traverse a subtree from 'parent' to count the number of descendent lineages
    and identify subclade roots that have 1+ or 2+ mutations.

    :param parent: A node in a TreeSwift tree.
    :param rng: Numpy random generator.
    :return: (lineages, one_mutation_subclade_roots, two_mutation_subclade_roots, rng)
    """
    lineages = 0
    one_mutation_subclade_roots = []
    two_mutation_subclade_roots = []

    for child in parent.child_nodes():
        if child.is_leaf():
            lineages += 1
        else:
            # mutation rate: 29903 * 0.00092 * branch_length
            mutations = rng.poisson(N_SITES * CLOCK * child.get_edge_length())
            # attach any mutations to that node
            if mutations > 0:
                child.mutations = mutations
                lineages += 1
                one_mutation_subclade_roots.append(child)
                if mutations > 1:
                    two_mutation_subclade_roots.append(child)
            else:
                derived_lineages, derived_one, derived_two, rng = get_clade_info(child, rng)
                lineages += derived_lineages
                one_mutation_subclade_roots.extend(derived_one)
                two_mutation_subclade_roots.extend(derived_two)

    return lineages, one_mutation_subclade_roots, two_mutation_subclade_roots, rng


def analyze_clades(output_dir: str, rng: np.random.Generator):
    """
    Analyze the time tree by simulating mutations and checking for 'AB' or 'CC' topologies
    based on distribution of subclade sizes.

    :param output_dir: Directory containing transmission_network.txt, sample_times.txt, etc.
    :param rng: Numpy random generator.
    :return: (AB, CC)
             AB = 1 if the AB topology conditions are met, otherwise 0
             CC = 1 if the CC topology conditions are met, otherwise 0
    """
    # set default output
    AB = 0
    CC = 0

    # get tree
    tree_path = os.path.join(output_dir, "tree.time.nwk")
    tree = treeswift.read_tree_newick(tree_path)
    tree.suppress_unifurcations()

    # Identify MRCA
    leaf_labels = {leaf.get_label() for leaf in tree.traverse_leaves()}
    mrca_node = tree.mrca(leaf_labels)

    # start analysis from the MRCA
    lineages, one_mutation_roots, two_mutation_roots, rng = get_clade_info(mrca_node, rng)

    # AB scenario: basal polytomy of >=100 lineages with a two-mutation subclade polytomy of >=100 lineages, the subclade size being 30-70% of total
    # basal polytomy constraint
    if lineages >= 100:
        # two-mutation subclade polytomy constraint
        for subclade_root in two_mutation_roots:
            subclade_lineages, _, _, rng = get_clade_info(subclade_root, rng)
            if subclade_lineages >= 100:
                # relative size constraint
                total_size = sum(1 for _ in mrca_node.traverse_leaves())
                subclade_size = sum(1 for _ in subclade_root.traverse_leaves())
                ratio = subclade_size / total_size if total_size > 0 else 0
                if 0.3 <= ratio <= 0.7:
                    AB = 1
                    break

    # CC scenario: exactly two lineages with one-mutation subclades each >=100 lineages and 30-70% of total
    # two one-mutation clade constraint
    elif lineages == 2 and len(one_mutation_roots) == 2:
        first_lineages, _, _, rng = get_clade_info(one_mutation_roots[0], rng)
        # polytomy constraints
        if first_lineages >= 100:
            second_lineages, _, _, rng = get_clade_info(one_mutation_roots[1], rng)
            if second_lineages >= 100:
                # relative size constraint
                total_size = sum(1 for _ in mrca_node.traverse_leaves())
                subclade_size = sum(1 for _ in one_mutation_roots[0].traverse_leaves())
                ratio = subclade_size / total_size if total_size > 0 else 0
                if 0.3 <= ratio <= 0.7:
                    CC = 1

    return AB, CC, tree


def compute_tree_hash(root: treeswift.Node):
    """
    Computes a SHA-256 hash of the tree branch lengths and the `mutations`
    attribute of each node, if present.
    """
    data_str = []
    for node in root.traverse_preorder():
        edge_length = node.get_edge_length() or 0.0
        mutations = getattr(node, "mutations", None)
        data_str.append(f"{edge_length};{mutations}")
    return hashlib.sha256("\n".join(data_str).encode("utf-8")).hexdigest()
