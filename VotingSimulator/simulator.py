# Contains logic for simulations

#TODO:
# Instantiate an appropriate number of agents via sampling or random variable creation
# Log the population somewhere
# Create random weights
# Predict each agents vote
# IF fraud is TRUE, do fraud and take note of that fraud
# Output the results of this election

import argparse
import sys
import json
import numpy as np
from tokenize import String
from agent import voting_agent
from nn import TinyModel

class simulation():
    def __init__(self, random_sampling, pop_size, sample_infile):
        self.random_sampling = random_sampling # True if the agents are sampled randomly from an existing population dataset
        self.pop_size = pop_size # The number of agents in the population
        self.sample_infile = sample_infile # The file name for the input file with agent variable names and parameters
        self.agent_list = self.create_agent_list() # A list containing all the agents (agent class)
        self.precincts = self.create_precincts_list() # A list of precincts attributes and results
        print(self.agent_list)
    
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
        print("Agent Attributes: [name, distribution_type, mean, std] ... ")
        for att in data:
            #[name, distribution_type, mean/start, std/end]
            print(att)

        # Agent VOTES on a candidate. vote is the key in attribute_dict.
        # TODO: voting mechanism. Do the vote and add some randomness at the end
        vote_prediction_model = TinyModel(input_size=len(data), output_size=2)
        for i in range(self.pop_size):
            agent = voting_agent(data)
            agent.attribute_dict["vote"] = vote_prediction_model.predict(list(agent.attribute_dict.values()))
            agent_list.append(agent)
            print(agent_list[i].attribute_dict)
            

        return agent_list

    def create_precincts_list(self):
        return []

def main():
    parser = argparse.ArgumentParser(description='Run a simulation')
    parser.add_argument('--random_sampling', type=bool, default=False, help='True if random samping should be used to create the population')
    parser.add_argument('--pop_size', type=int, default=100, help='number of agents in the population')
    parser.add_argument('--sample_infile', type=str, default="agent_vars.json")
    args = parser.parse_args()
    args_dict = vars(args)

    # Create a simulation
    simy = simulation(args_dict["random_sampling"], args_dict["pop_size"], args_dict["sample_infile"])

if __name__ == "__main__":
    main()