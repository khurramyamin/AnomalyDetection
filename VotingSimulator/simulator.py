# Contains logic for simulations

#TODO:
# Instantiate an appropriate number of agents via sampling or random variable creation
# Log the population somewhere
# Create random weights
# Predict each agents vote
# IF fraud is TRUE, do fraud and take note of that fraud
# Output the results of this election

import argparse
from re import X
import sys
import json
import numpy as np
from tokenize import String

from torch import dropout
from agent import voting_agent
from nn import TinyModel
import matplotlib.pyplot as plt

class simulation():
    def __init__(self, random_sampling, pop_size, sample_infile, randomness_vars, NN = None):
        self.trues = 0
        self.NN = NN
        self.random_sampling = random_sampling # True if the agents are sampled randomly from an existing population dataset
        self.pop_size = pop_size # The number of agents in the population
        self.sample_infile = sample_infile # The file name for the input file with agent variable names and parameters
        self.include_dropout = randomness_vars[4] # True if should include dropout
        self.dropout_prob = randomness_vars[5] # prob of a dropout
        self.agent_list = self.create_agent_list() # A list containing all the agents (agent class)

        if (randomness_vars[0]):
            self.add_result_randomness(randomness_vars[1])
        if (randomness_vars[2]):
            self.add_result_randomness(randomness_vars[3])

    def read_agent_attribute(self):
        file = open(self.sample_infile)
        data = json.load(file)
        for att in data:
            #[name, distribution_type, mean, std]
            print(att)

    def create_agent_list(self):
        agent_list = []
        file = open(self.sample_infile)
        data = json.load(file)
        # print("Agent Attributes: [name, distribution_type, mean, std] ... ")
        # for att in data:
        #     #[name, distribution_type, mean/start, std/end]
        #     print(att)

        # Agent VOTES on a candidate. vote is the key in attribute_dict.
        # TODO: voting mechanism. Do the vote and add some randomness at the end
        if(self.NN == None):
            vote_prediction_model = TinyModel(input_size=len(data), output_size=1, dropout_included=self.include_dropout, dropout_prob=self.dropout_prob)
        else:
            vote_prediction_model = self.NN
        votes_values = np.empty((self.pop_size,))
        for i in range(self.pop_size):
            agent = voting_agent(data)
            normalized_attr_list = self.normalize_attr(list(agent.attribute_dict.values()))
            votes_values[i] = vote_prediction_model.predict(normalized_attr_list)
            agent_list.append(agent)
        #print(votes_values)
        median = np.median(votes_values) #TODO: Figure out way to make median work better with many precincts
        #print(median)

        count_true = 0 #
        count_false = 0 #

        for i in range(self.pop_size):
            agent = agent_list[i]
            if(votes_values[i] < median):
                agent.attribute_dict["vote"] = True
                count_true += 1
            elif(votes_values[i] == median):
                if np.random.randint(2) == 1:
                    agent.attribute_dict["vote"] = True
                    count_true += 1
                else:
                    agent.attribute_dict["vote"] = False
                    count_false += 1
            else:
                agent.attribute_dict["vote"] = False
                count_false += 1
            agent_list[i] = agent
            # print(agent_list[i].attribute_dict)

        #print("TRUES: ", count_true) #
        self.trues = count_true
        #print("FALSES: ", count_false) #

        return agent_list

    def visualize(self):
        x = []
        y = []
        colors = []
        for i in range(self.pop_size):
            agent = self.agent_list[i]
            x.append(agent.attribute_dict["x"])
            y.append(agent.attribute_dict["y"])
            if (agent.attribute_dict["vote"]):
                colors.append((1,0,0))
            else:
                colors.append((0,1,0))
        plt.scatter(x, y, c=colors, s=10)

        plt.show()
        return

    def normalize_attr(self, list_):
        return list_

    def add_result_randomness(self, rand_amount):
        for i in range(self.pop_size):
            if(np.random.uniform(0,1) < rand_amount):
                agent = self.agent_list[i]
                agent.attribute_dict["vote"] = not agent.attribute_dict["vote"]
                self.agent_list[i] = agent
    
    def results(self):
        result = []
        all_votes_candidates = []
        all_votes_attributes = []
        keys_list = list(self.agent_list[0].attribute_dict.keys())
        for i in range(len(keys_list)-1):
            total = 0
            for j in range(self.pop_size):
                agent = self.agent_list[j]
                total += agent.attribute_dict[keys_list[i]]
            result.append(total/self.pop_size)
        for j in range(self.pop_size):
            agent = self.agent_list[j]

            agent_attribs = []
            for var_keys in agent.attribute_dict.keys():
                if agent.attribute_dict[var_keys] == 1:
                    if (var_keys != "vote"):
                        agent_attribs.append(var_keys)
            all_votes_candidates.append(agent.attribute_dict["vote"])
            all_votes_attributes.append(agent_attribs)
        result.append(self.trues/self.pop_size)
        return result, all_votes_candidates, all_votes_attributes


def run_sim(
    NeuralNetwork, random_sampling: bool = False, pop_size: int = 100, sample_infile: str = "agent_vars.json", visualize: bool = False,
    result_flip_randomness: bool = False, result_flip_randomness_amount: float = 0,
    result_threshold_randomness: bool = False, result_threshold_randomness_amount: float = 0,
    result_dropout_randomness: bool = False, result_dropout_randomness_amount: float = 0,
    return_results_list: bool = True
    ):

    # Create a simulation
    randomness_vars = [result_flip_randomness, result_flip_randomness_amount, result_threshold_randomness, result_threshold_randomness_amount, result_dropout_randomness, result_dropout_randomness_amount]
    simy = simulation(random_sampling, pop_size, sample_infile, randomness_vars, NeuralNetwork)

    if(visualize):
        simy.visualize()

    if(return_results_list):
        results = simy.results()
        return results
    return None

def main():
    parser = argparse.ArgumentParser(description='Run a simulation')
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
    args = parser.parse_args()
    args_dict = vars(args)

    # Create a simulation
    randomness_vars = [args_dict["result_flip_randomness"], args_dict["result_flip_randomness_amount"], args_dict["result_threshold_randomness"], args_dict["result_threshold_randomness_amount"], args_dict["result_dropout_randomness"], args_dict["result_dropout_randomness_amount"]]
    simy = simulation(args_dict["random_sampling"], args_dict["pop_size"], args_dict["sample_infile"], randomness_vars)
    
    print(args_dict)

    if(args_dict["visualize"]):
        simy.visualize()

    if(args_dict["return_results_list"]):
        results, all_votes = simy.results()
        return results, all_votes

if __name__ == "__main__":
    main()