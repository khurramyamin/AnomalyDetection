# Contains code to run one or many instances of the simulator
# takes in parameters like whether there should be fraud, where to save the files, where the population should be sampled from (or if its completely random), and an example agent_attribute.json file

import argparse
import sys
import json
from typing import Tuple
import simulator
import numpy as np
from nn import TinyModel


def main():
    parser = argparse.ArgumentParser(description='Run a simulation')
    parser.add_argument('--precinct_num', type=int, default=1, help='Precincts')
    parser.add_argument('--random_sampling', type=bool, default=False, help='True if random samping should be used to create the population')
    parser.add_argument('--pop_size', type=int, default=100000, help='number of agents in the population')
    parser.add_argument('--sample_infile', type=str, default="agent_vars.json", help='json file with a list of different agent attributes')
    parser.add_argument('--visualize', type=bool, default=False, help = "whether to visualize the agent results")
    parser.add_argument('--result_flip_randomness', type=bool, default=True, help = "whether to add flip randomness to the results. Flips person's vote")
    parser.add_argument('--result_flip_randomness_amount', type=float, default=.3, help = "How likely flipping is, from 0 to 1.")
    parser.add_argument('--result_threshold_randomness', type=bool, default=True, help = "whether to add threshold randomness to the results. Closer to threshold more effected by this noise")
    parser.add_argument('--result_threshold_randomness_amount', type=float, default=.2, help = "how much threshold randomness to add. Between 0 and 1.")
    parser.add_argument('--result_dropout_randomness', type=bool, default=True, help = "whether to add dropout randomness to the results. Uses pytorch dropout functionality")
    parser.add_argument('--result_dropout_randomness_amount', type=float, default=.2, help = "how much dropout randomness to add. Between 0 and 1.")
    parser.add_argument('--return_results_list', type=bool, default=True, help = "Whether to return a list with all the results")
    parser.add_argument('--deletion_fraud', type=bool, default=False, help = "Whether to delete votes of a certain random candidate")
    parser.add_argument('--deletion_fraud_amount', type=float, default=0, help = "how much deletion fraud. Between 0 and 1.")
    parser.add_argument('--addition_fraud', type=bool, default=False, help = "Whether to add votes for a certain random candidate")
    parser.add_argument('--addition_fraud_amount', type=float, default=0, help = "how much addition fraud. Between 0 and 1.")
    parser.add_argument('--swapping_fraud', type=bool, default=False, help = "Whether to swap votes in favor of a certain random candidate")
    parser.add_argument('--swapping_fraud_amount', type=float, default=0, help = "how much swapping fraud. Between 0 and 1.")

    args_ = parser.parse_args()
    args = vars(args_)

    file = open(args["sample_infile"])
    data = json.load(file)
    NN = TinyModel(input_size=len(data), output_size=1, dropout_included=args["result_dropout_randomness"], dropout_prob=args["result_dropout_randomness_amount"])

    total_precincts_arr = []

    all_precinct_poll = []
    all_precinct_attribs = []
    all_precincts_covar = []
    for i in range(args["precinct_num"]):
        precinct_arr, all_votes_candidates, all_votes_attribs = simulator.run_sim(NN, args["random_sampling"], args["pop_size"], args["sample_infile"], args["visualize"], args["result_flip_randomness"], args["result_flip_randomness_amount"], args["result_threshold_randomness"], args["result_threshold_randomness_amount"], args["result_dropout_randomness"], args["result_dropout_randomness_amount"], args["return_results_list"])
        total_precincts_arr.append(precinct_arr)
        poll_result_candidates, poll_result_attribs = poll(all_votes_candidates, all_votes_attribs, .1)
        all_precinct_poll.append(poll_result_candidates)
        all_precinct_attribs.append(np.array(poll_result_attribs).flatten())
        all_precincts_covar.append(np.array(poll_result_attribs))
    all_precinct_poll = np.array(all_precinct_poll).flatten()
    all_precinct_attribs = np.array(all_precinct_attribs).flatten()
    poll_result = np.count_nonzero(all_precinct_poll == True) / all_precinct_poll.shape[0]

    covar_attribs = np.array(all_precincts_covar).reshape((-1,2))
    unique_covar_attribs, covar_counts = np.unique(covar_attribs, return_counts=True, axis=0)
    poll_covar_dicy = {}
    poll_covar_dicy_a = {}
    poll_covar_dicy_b = {}

    for i, covar in enumerate(unique_covar_attribs):
        poll_covar_dicy[(tuple(covar))] = covar_counts[i]
        a_index = np.where(all_precinct_poll == True)
        poll_covar_dicy_a[(tuple(covar))] = np.count_nonzero(np.array(covar_attribs)[a_index] == covar)
        b_index = np.where(all_precinct_poll == False)
        poll_covar_dicy_b[(tuple(covar))] = np.count_nonzero(np.array(covar_attribs)[b_index] == covar)


    uniques, counts = np.unique(all_precinct_attribs, return_counts=True)
    all_percentages = dict(zip(uniques, counts * 100 / len(all_precinct_attribs)))

    a_cand_index = np.where(all_precinct_poll == True)
    b_cand_index = np.where(all_precinct_poll == False)
    print(a_cand_index)
    print(b_cand_index)

    candidate_a_attribs = all_precinct_attribs[a_cand_index]
    uniques, counts = np.unique(candidate_a_attribs, return_counts=True)
    a_counts = dict(zip(uniques, counts)) #a_percentages = dict(zip(uniques, counts * 100 / len(candidate_a_attribs)))

    print(uniques)

    candidate_b_attribs = all_precinct_attribs[b_cand_index]
    uniques, counts = np.unique(candidate_b_attribs, return_counts=True)
    b_counts = dict(zip(uniques, counts)) #b_percentages = dict(zip(uniques, counts * 100 / len(candidate_b_attribs)))

    print(uniques)

    # NOTE: total_precincts_arr is the array with all the precincts data. e.g., [[attrib1_avg, attrib2_avg, vote_1], [attrib1_avg, attrib2_avg, vote_2], [attrib1_avg, attrib2_avg, vote_3]]
    #print(total_precincts_arr)

    # NOTE: poll_results is the final candidate results of the poll
    print(poll_result)

    # NOTE: These are a dictionary with percentages of the poll
    #print(all_percentages)
    print(a_counts)
    print(b_counts)

    # NOTE: For these the results won't have 100% of the attributes if they never appear, so just reference agent_vars instead for now

    #This is poll covariance dictionary
    
    #print(poll_covar_dicy)

    print("a ", poll_covar_dicy_a)

    print("b ", poll_covar_dicy_b)


#THIS IS WHERE FRAUD INJECTION WILL BE WRITTEN










def poll(all_votes_candidates : list, all_votes_attribs : list, percentage : int):
    length_ = len(all_votes_candidates)
    index_arr = np.random.randint(low = 0, high = length_, size = (int(length_ * percentage),))
    poll_result_candidates = np.array(all_votes_candidates)[index_arr]
    poll_result_attribs = np.array(all_votes_attribs)[index_arr]
    return poll_result_candidates, poll_result_attribs


if __name__ == "__main__":
    main()

#add positional encoding TODO