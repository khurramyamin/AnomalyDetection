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


def main():
    parser = argparse.ArgumentParser(description='Run a simulation')
    parser.add_argument('--pop_size', type=int, default=100, help='number of agents in the population')
    parser.add_argument('--random_sampling', type=bool, default=False, help='True if random samping should be used to create the population')
    parser.add_argument('--sample_file', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument('--outfile', nargs='?', type=argparse.FileType('w'), default=sys.stdout)
    


    args = parser.parse_args()

if __name__ == "__main__":
    main()