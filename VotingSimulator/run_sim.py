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
    parser.add_argument('--precinct_num', type=int, default=100, help='Precincts')
    parser.add_argument('--random_sampling', type=bool, default=False, help='True if random samping should be used to create the population')
    parser.add_argument('--pop_size', type=int, default=100, help='number of agents in the population')
    parser.add_argument('--sample_infile', type=str, default="agent_vars.json", help='json file with a list of different agent attributes')
    parser.add_argument('--visualize', type=bool, default=False, help = "whether to visualize the agent results")
    parser.add_argument('--result_flip_randomness', type=bool, default=False, help = "whether to add flip randomness to the results. Flips person's vote")
    parser.add_argument('--result_flip_randomness_amount', type=float, default=0, help = "How likely flipping is, from 0 to 1.")
    parser.add_argument('--result_threshold_randomness', type=bool, default=False, help = "whether to add threshold randomness to the results. Closer to threshold more effected by this noise")
    parser.add_argument('--result_threshold_randomness_amount', type=float, default=0, help = "how much threshold randomness to add. Between 0 and 1.")
    parser.add_argument('--result_dropout_randomness', type=bool, default=False, help = "whether to add dropout randomness to the results. Uses pytorch dropout functionality")
    parser.add_argument('--result_dropout_randomness_amount', type=float, default=0, help = "how much dropout randomness to add. Between 0 and 1.")
    parser.add_argument('--return_results_list', type=bool, default=True, help = "Whether to return a list with all the results")
    args_ = parser.parse_args()
    args = vars(args_)

    file = open(args["sample_infile"])
    data = json.load(file)
    NN = TinyModel(input_size=len(data), output_size=1, dropout_included=args["result_dropout_randomness"], dropout_prob=args["result_dropout_randomness_amount"])

    total_precincts_arr = []

    all_precinct_poll = []
    for i in range(args["precinct_num"]):
        precinct_arr, all_votes = simulator.run_sim(NN, args["random_sampling"], args["pop_size"], args["sample_infile"], args["visualize"], args["result_flip_randomness"], args["result_flip_randomness_amount"], args["result_threshold_randomness"], args["result_threshold_randomness_amount"], args["result_dropout_randomness"], args["result_dropout_randomness_amount"], args["return_results_list"])
        total_precincts_arr.append(precinct_arr)
        poll_result = poll(all_votes, .1)
        all_precinct_poll.append(poll_result)


    all_precinct_poll = [item for sublist in all_precinct_poll for item in sublist]
    poll_result = all_precinct_poll.count(True) / len(all_precinct_poll)

    # NOTE: This is the array with all the precincts data. e.g., [[attrib1_avg, attrib2_avg, vote_1], [attrib1_avg, attrib2_avg, vote_2], [attrib1_avg, attrib2_avg, vote_3]]
    print(total_precincts_arr)

    print(poll_result)






def poll(all_vote : list, percentage : int):
    poll_result = np.random.choice(all_vote, size = int(len(all_vote) * percentage))
    return poll_result


if __name__ == "__main__":
    main()

#add positional encoding TODO