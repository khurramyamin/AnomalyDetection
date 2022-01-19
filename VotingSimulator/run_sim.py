# Contains code to run one or many instances of the simulator
# takes in parameters like whether there should be fraud, where to save the files, where the population should be sampled from (or if its completely random), and an example agent_attribute.json file

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description='Run a simulation')
    parser.add_argument('--sim_iterations', type=int, default=1, help='number of simulations ran')
    parser.add_argument('--pop_size', type=int, default=100, help='number of agents in the population')
    parser.add_argument('--random_sampling', type=bool, default=False, help='True if random samping should be used to create the population')
    parser.add_argument('--sample_file', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument('--outfile', nargs='?', type=argparse.FileType('w'), default=sys.stdout)
    


    args = parser.parse_args()

if __name__ == "__main__":
    main()