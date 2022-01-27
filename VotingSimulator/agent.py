# Contains agent class, each agent is an individual in the environment
# has attributes, attributes are based off an agent_attributes.json file (name may differ)
from mimetypes import init
import json
import numpy as np

class voting_agent():
    # init method
    def __init__(self, agent_vars):
        self.attribute_dict = self.create_attributes_dict(agent_vars)

    def create_attributes_dict(self, agent_vars):
        attribute_dict = {}

        # Loop through all variables from the json file and randomly initialize them 
        for var in agent_vars:
            if (var[1] == "normal_dist"):
                value = np.random.normal(loc=var[2], scale=var[3], size=None)
            elif (var[1] == "uniform_dist"):
                value = np.random.uniform(low=var[2], high=var[3], size=None)
            elif (var[1] == "uniform_int_dist"):
                value = np.random.randint(low=var[2], high=var[3], size=None, dtype=int)
            attribute_dict[var[0]] = value
        return attribute_dict