import pandas as pd
import numpy as np
from sim import econ, plot, simulation


def run_sim(total_available):
    '''
    Input: total available amount of water per capita in acre-feet per year

    Output: Plots of optimal policies in output/scenarios
    '''

    # Setting parameters for the simulation
    population_size = 5000
    total_available *= population_size 

    thresholds = np.arange(0,10,0.5)        # this defines the tresholds between prices in acre-feet per household
    price_1_list = range(1000, 4000,100)    # this defines the price charged below the threshold
    price_2_list = range(1000, 4000,100)    # this defines the price charged above the threshold

    # running simulation
    metrics = simulation.simulation(price_1_list, price_2_list, thresholds, population_size, total_available)

    # plotting results
    plot.plot_simulation(metrics)


# in case the simulation is run separately
if __name__=='__main__':
    run_sim(0.4124291510893835)