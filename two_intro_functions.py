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
import treeswift

from shared_functions import *

TMRCA_TO_INTRO_VALUES = [0, 5, 10, 15, 20, 25, 30]


def get_sim_data(simulation_id, suffix):
    """
    Read transmission network and disease details from file to lists, appending a suffix to the ID's of all nodes in the data.

    :param simulation_id: the number of the one intro simulation directory to read the files from.
    :param suffix: '_0' or '_1' (in order to differentiate the IDs of nodes from the different introductions).
    """
    with open(os.path.join("one_intro", str(simulation_id), "transmission_network.txt"), 'r') as file:
        transmission_network = []
        for line in file:
            parts = line.strip().split('\t')
            parts[0] = ("Upstream_lineage" if parts[0] == "None" else parts[0]) + suffix
            parts[1] = parts[1] + suffix
            parts[2] = float(parts[2])  # Cast the third string to float
            transmission_network.append(parts)
    with open(os.path.join("one_intro", str(simulation_id), "disease_events.txt"), 'r') as file:
        disease_events = []
        for line in file:
            parts = line.strip().split(' ')
            parts[0] = float(parts[0])  # Cast the first string to float
            parts[1] = parts[1] + suffix
            disease_events.append(parts)
    return transmission_network, disease_events


def write_transmission_network(output_dir, transmission_network_0, transmission_network_1, offset_0, offset_1, sim_end_time):
    """
    Write offset, merged transmissions events to file, stopping after 50,000.

    :param output_dir: Directory for outputs.
    :param transmission_network_0: List of transmission events from one introduction
    :param transmission_network_1: List of transmission events from the other introduction
    :param offset_0: Time to offset '_0' introduction event times by, in years.
    :param offset_1: Time to offset '_1' introduction event times by, in years.
    :param sim_end_time: Maximum time for simulated transmission events.
    :return first_50k_infections: set of nodes infected by the firsty 50k transmissions
    """

    # start indexes for each list and record their maxima
    i_0, i_1 = 0, 0
    len_0, len_1 = len(transmission_network_0), len(transmission_network_1)

    count_written = 0 # increment and stop when this reaches 50k
    first_50k_infections = set() # record these to return
    with open(os.path.join(output_dir, "transmission_network.txt"), 'w') as file:
        # add the MRCA two upstream lineages
        file.write('None\tMRCA\t0.0\n')
        file.write('MRCA\tUpstream_lineage_0\t0.0\n')
        file.write('MRCA\tUpstream_lineage_1\t0.0\n')
        # go through both lists, incrementing the index as entries are written until one list ends
        while i_0 < len_0 and i_1 < len_1 and count_written < 50000:
            t_0 = transmission_network_0[i_0][2] + offset_0
            t_1 = transmission_network_1[i_1][2] + offset_1
            if min(t_0, t_1) >= sim_end_time: # stop if end time is exceeded
                return first_50k_infections
            # write the earliest event from the two lists
            if t_0 <= t_1:
                file.write(transmission_network_0[i_0][0] + '\t' + transmission_network_0[i_0][1] + '\t' + f'{t_0:.6f}\n')
                first_50k_infections.add(transmission_network_0[i_0][1])
                i_0 += 1
            else:
                file.write(transmission_network_1[i_1][0] + '\t' + transmission_network_1[i_1][1] + '\t' + f'{t_1:.6f}\n')
                first_50k_infections.add(transmission_network_1[i_1][1])
                i_1 += 1
            count_written += 1
        # go through the remaining list
        while i_0 < len_0 and count_written < 50000:
            t_0 = transmission_network_0[i_0][2] + offset_0
            if t_0 >= sim_end_time:
                return first_50k_infections
            file.write(transmission_network_0[i_0][0] + '\t' + transmission_network_0[i_0][1] + '\t' + f'{t_0:.6f}\n')
            first_50k_infections.add(transmission_network_0[i_0][1])
            i_0 += 1
            count_written += 1
        while i_1 < len_1 and count_written < 50000:
            t_1 = transmission_network_1[i_1][2] + offset_1
            if t_1 >= sim_end_time:
                return first_50k_infections
            file.write(transmission_network_1[i_1][0] + '\t' + transmission_network_1[i_1][1] + '\t' + f'{t_1:.6f}\n')
            first_50k_infections.add(transmission_network_1[i_1][1])
            i_1 += 1
            count_written += 1

    return first_50k_infections


def write_disease_events(output_dir, disease_events_0, disease_events_1, offset_0, offset_1, first_50k_infections, sim_end_time):
    """
    Write offset, merged disease events to file, if they are amongst the first 50k.

    :param output_dir: Directory for outputs.
    :param disease_events_0: List of disease events from one introduction
    :param disease_events_1: List of disease events from the other introduction
    :param offset_0: Time to offset '_0' introduction event times by, in years.
    :param offset_1: Time to offset '_1' introduction event times by, in years.
    :param first_50k_infections: set of nodes infected by the firsty 50k transmissions
    :param sim_end_time: Maximum time for simulated transmission events.
    """

    # start indexes for each list and record their maxima
    i_0, i_1 = 0, 0
    len_0, len_1 = len(disease_events_0), len(disease_events_1)
    with open(os.path.join(output_dir, "disease_events.txt"), 'w') as file:
        # go through both lists, incrementing the index as entries are written until one list ends
        while i_0 < len_0 and i_1 < len_1:
            t_0 = disease_events_0[i_0][0] + offset_0
            t_1 = disease_events_1[i_1][0] + offset_1
            if min(t_0, t_1) >= sim_end_time: # stop if end time is exceeded
                return
            # write the earliest event from the two lists
            if t_0 <= t_1:
                if disease_events_0[i_0][1] in first_50k_infections:
                    file.write(f'{t_0:.6f} ' + disease_events_0[i_0][1] + ' ' + disease_events_0[i_0][2] + ' ' + disease_events_0[i_0][3] + '\n')
                i_0 += 1
            else:
                if disease_events_1[i_1][1] in first_50k_infections:
                    file.write(f'{t_1:.6f} ' + disease_events_1[i_1][1] + ' ' + disease_events_1[i_1][2] + ' ' + disease_events_1[i_1][3] + '\n')
                i_1 += 1
        # go through the remaining list
        while i_0 < len_0 :
            t_0 = disease_events_0[i_0][0] + offset_0
            if t_0 >= sim_end_time:
                return
            if disease_events_0[i_0][1] in first_50k_infections:
                file.write(f'{t_0:.6f} ' + disease_events_0[i_0][1] + ' ' + disease_events_0[i_0][2] + ' ' + disease_events_0[i_0][3] + '\n')
            i_0 += 1
        while i_1 < len_1:
            t_1 = disease_events_1[i_1][0] + offset_1
            if t_1 >= sim_end_time:
                return
            if  disease_events_1[i_1][1] in first_50k_infections:
                file.write(f'{t_1:.6f} ' + disease_events_1[i_1][1] + ' ' + disease_events_1[i_1][2] + ' ' + disease_events_1[i_1][3] + '\n')
            i_1 += 1

def get_lineages(parent: treeswift.Node):
    """
    Recursively traverse a subtree from 'parent' to count the number of descendent lineages.

    :param parent: A node in a TreeSwift tree.
    :return: The number of descendent lineages.
    """
    lineages = 0

    for child in parent.child_nodes():
        if child.is_leaf():
            lineages += 1
        elif hasattr(child, "mutations"):
            lineages += 1
        else:
            lineages += get_lineages(child)

    return lineages

def check_arises(AB, CC, output_dir, tree):
    """
    :param AB, CC: 1 if topology is detected, 0 if not
    :param output_dir: working directory
    :param tree: tree with mutations attached at nodes
    :return ABB, CC: As received, unless a test is failed, then return 0, 0
    """

    # check relative size
    size_0 = 0
    size_1 = 0
    with open(os.path.join(output_dir, "sample_times.txt")) as f:
        for line in f:
            fields = line.split('\t')
            if fields[0].endswith('_0'):
                size_0 += 1
            elif fields[0].endswith('_1'):
                size_1 += 1
    if size_0+size_1 == 0:
        return 0, 0
    if not (0.3 <= size_0/(size_0+size_1) <= 0.7):
        return 0, 0

    # check separation
    leaf_labels = {leaf.get_label() for leaf in tree.traverse_leaves()}
    mrca_node = tree.mrca(leaf_labels)
    separation = 0
    for intro in mrca_node.child_nodes():
        if hasattr(intro, "mutations"):
            separation += intro.mutations
    if separation < 2:
        return 0, 0

    # check polytomies
    for intro in mrca_node.child_nodes():
        lineages = get_lineages(intro)
        if hasattr(intro, "mutations"):
            if lineages < 100: # subclade
                return 0, 0
        elif lineages < 99: #basal clade needs to count the other clade
            return 0, 0
    # all passed?
    return AB, CC


def two_intro(output_dir, parameters_file_name, rng, id_0, id_1):
    """
    Sample two one introduction simulations. Combine and analyze for a range of possible introduction timings.

    :param output_dir: Directory for  outputs.
    :param parameters_file_name: The GEMF parameters file.
    :param rng: Numpy random generator.
    :param id_0: id of one simulation
    :param id_1: id of anothersimulation

    """
    os.makedirs(output_dir, exist_ok=True)
    sim_end_time = get_sim_end_time(parameters_file_name)

    # Draw two one intro simulations and fetch data
    transmission_network_0, disease_events_0 = get_sim_data(id_0, "_0")
    transmission_network_1, disease_events_1 = get_sim_data(id_1, "_1")

    # analyse possible combinations of times t0, t1 between MRCA and introductions 1, 2
    # Create integer arrays for AB and CC
    AB_array = np.zeros((len(TMRCA_TO_INTRO_VALUES), len(TMRCA_TO_INTRO_VALUES)),dtype=int)
    CC_array = np.zeros((len(TMRCA_TO_INTRO_VALUES), len(TMRCA_TO_INTRO_VALUES)),dtype=int)
    for i0, t0 in enumerate(TMRCA_TO_INTRO_VALUES):
        for i1, t1 in enumerate(TMRCA_TO_INTRO_VALUES):

            offset_0 = t0/365 # convert to years
            offset_1 = t1/365

            # 1. Generate combined simulation data
            adjusted_sim_end_time = sim_end_time + min(offset_0, offset_1)
            first_50k_infections = write_transmission_network(output_dir, transmission_network_0, transmission_network_1, offset_0, offset_1, adjusted_sim_end_time)
            write_disease_events(output_dir, disease_events_0, disease_events_1, offset_0, offset_1, first_50k_infections, adjusted_sim_end_time)

            # 2. Generate sample times
            rng = simulate_sequencing(output_dir, adjusted_sim_end_time, rng)

            # 3. Generate viral phylogeny
            rng = simulate_coalescence(output_dir, rng)

            # 4. Simulate mutations and analyze clades
            AB, CC, tree = analyze_clades(output_dir, rng)

            # 5. Check if the two clades arise from the separate introductions
            if AB or CC:
                AB, CC = check_arises(AB, CC, output_dir, tree)

            AB_array[i0, i1] += AB
            CC_array[i0, i1] += CC

    # Write AB to file
    ab_path = os.path.join(output_dir, 'AB_array.csv')
    np.savetxt(ab_path, AB_array, delimiter=",", fmt='%d')

    # Write CC to file
    cc_path = os.path.join(output_dir, 'CC_array.csv')
    np.savetxt(cc_path, CC_array, delimiter=",", fmt='%d')

    # store the final tree
    tree.write_tree_newick(os.path.join(output_dir, "final_tree.nwk"))

    # store a hash of the final tree, for verification
    output_hash = compute_tree_hash(tree.root)
    with open(os.path.join(output_dir, 'final_tree_hash.txt'), 'w') as f:
        f.write(output_hash)

