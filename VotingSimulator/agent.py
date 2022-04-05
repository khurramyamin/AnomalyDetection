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

        currAttribsArr = []
        # Loop through all variables from the json file and randomly initialize them 
        for i, var in enumerate(agent_vars):
            if (i + 1 < len(agent_vars) and agent_vars[i + 1][0] == var[0]):
                currAttribsArr.append(i)
            else :
                currAttribsArr.append(i)
                if (var[4] == "%"):
                    value = np.random.randint(low=0, high=101)
                    currVal = 0
                    found = False
                    for index in currAttribsArr:
                        currVal += agent_vars[index][2]
                        if not found and currVal >= value:
                            if var[3] == "seperate":
                                attribute_dict[agent_vars[index][1]] = 1
                            elif var[3] == "combined":
                                attribute_dict[agent_vars[index][0]] = agent_vars[index][1]
                            found = True
                        elif (var[3] == "seperate"):
                            attribute_dict[agent_vars[index][1]] = 0
                currAttribsArr = []
            

            
            # if (var[1] == "normal_dist"):
            #     value = np.random.normal(loc=var[2], scale=var[3], size=None)
            # elif (var[1] == "uniform_dist"):
            #     value = np.random.uniform(low=var[2], high=var[3], size=None)
            # elif (var[1] == "uniform_int_dist"):
            #     value = np.random.randint(low=var[2], high=var[3], size=None, dtype=int)
            # attribute_dict[var[0]] = value
        return attribute_dict