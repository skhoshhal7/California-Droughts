import numpy as np
import pandas as pd


def generate_scenario(population_size):
    '''
    Sets parameters according to our economic analysis and literature review.
    Creates a dataframe of individuals
    '''
    # print(" > Generating scenario...")
    population = population_size    # size of our town
    household_size = 4              # number of people per household
    avg_demand_elasticity = -0.435  # in $/acre-feet of water
    avg_use = 100                   # in gallons per day per person
    avg_water_cost = 77             # in $ per month per household of 4
    median_income = 78672           # for population
    sigma_income = 0.73796
    
    
    # transform parameters
    yearly_use = avg_use * 365.25 * household_size          # yearly use in gallons per household
    yearly_use_af = yearly_use / 325851                     # yearly use in acre-feet per household
    yearly_cost = avg_water_cost * 12 * household_size/ 4   # yearly cost in $ per household
    current_cost_per_af = yearly_cost/yearly_use_af

    # other assumptions
    part_of_use_that_is_fixed = 2/3 # https://www.google.com/search?q=water+usage+by+income&client=safari&rls=en&sxsrf=ALiCzsbcJZmmS70aTZQlw8zcg9E0IwfbUA:1670270330064&source=lnms&tbm=isch&sa=X&ved=2ahUKEwiI7M7toeP7AhVXmHIEHRW1CE8Q_AUoAXoECAEQAw&biw=1440&bih=812&dpr=2#imgrc=g1km3zfEH6yM7M

    # generate individuals
    income = np.random.lognormal(mean=np.log(median_income), sigma=sigma_income, size=population)    # https://statisticsbyjim.com/probability/global-income-distributions/
    income_quintile = pd.qcut(income,5, labels=range(1,6))
    current_use =  part_of_use_that_is_fixed * yearly_use_af + (((1-part_of_use_that_is_fixed)*yearly_use_af)/(median_income)) * income   # https://www.researchgate.net/figure/Monthly-household-income-vs-daily-water-consumption_fig3_325584808                                                     # https://en.wikipedia.org/wiki/Residential_water_use_in_the_U.S._and_Canada
    current_use += np.random.normal(0,current_use/10)
    current_cost = current_cost_per_af * current_use
    elasticity = avg_demand_elasticity + ((np.log(income)-np.log(median_income))/-10) # https://www.researchgate.net/figure/Estimated-price-elasticity-of-water-demand-for-low-middle-and-high-income-groups_tbl1_228917285
    
    population_df = pd.DataFrame({
        'income':income,
        'income_quintile':income_quintile,
        'current_use':current_use,
        'current_cost':current_cost,
        'elasticity':elasticity
    })
    return population_df
    


def simulate_demand(population_df, price1, price2, threshold):
    '''
    Input: price (in $ per acre-foot)
    Output: water usage (in acre-foot per year)
    '''
    # print(" > Simulating demand...")
    price_below_threshold = min(price1,price2)
    price_above_threshold = max(price1,price2)

    ## demand for lower price 
    population_df['theoretical_demand_lower_price'] = population_df['current_use'] * (price_below_threshold/population_df['current_cost']) ** population_df['elasticity']
    population_df['theoretical_demand_higher_price'] = population_df['current_use'] * (price_above_threshold/population_df['current_cost']) ** population_df['elasticity']
    population_df['threshold'] = threshold
    population_df['used_at_lower_price'] = population_df[['theoretical_demand_lower_price','threshold']].min(axis=1)
    
    ## demand for higher price
    population_df.loc[population_df['theoretical_demand_lower_price']<threshold,'used_at_higher_price'] = 0
    population_df.loc[population_df['theoretical_demand_higher_price']<threshold,'used_at_higher_price'] = 0
    population_df.loc[population_df['theoretical_demand_higher_price']>=threshold,'used_at_higher_price'] = population_df['theoretical_demand_higher_price'] - threshold

    ## total expenses
    population_df['total_used'] = population_df['used_at_lower_price'] + population_df['used_at_higher_price'] 
    population_df['total_spent'] = population_df['used_at_lower_price'] * price_below_threshold + population_df['used_at_higher_price'] * price_above_threshold
    return population_df.drop('threshold',axis=1)