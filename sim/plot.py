import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context("paper", font_scale = 1.2)
color = sns.color_palette("tab10", 10)[0]

def is_in_frontier(df, metric1, metric2, row):
    '''
    returns a flag on whether a point is in the efficient frontier in a scatterplot.
    '''
    df_filtered = df.loc[((df[metric1] < df.loc[row,metric1]) & (df[metric2] < df.loc[row, metric2])),:].copy()
    if len(df_filtered) == 0:
        return True
    else:
        return False

def plot_simulation(metrics):
    '''
    Saves five different plots in output/scenarios
    '''

    print(" > Exporting plots...")
    print(" > Plot 1: What are the different simulations (demand curve)? Which of them are feasible?")
    metrics['total_costs'] /= 1000000
    sns.scatterplot(metrics, x = "total_costs", y = "total_use", hue = "feasible")
    plt.title('All scenarios')
    plt.xlabel('Total amount spent on water by population (in million USD)')
    plt.ylabel('Total amount used by population (in acre-feet)')
    plt.savefig('output/scenarios/1-all_scenarios.png',dpi=600)
    plt.clf()

    print(" > Plots: What is our menu of options? (equity vs. economic damage)")
    metrics_of_feasible = metrics.loc[metrics['feasible']==True]
    sns.scatterplot(metrics_of_feasible, x='total_costs', y='perc_spent_by_bottom_quintile', hue='total_use')
    plt.savefig('output/scenarios/2a-perc_spent_by_bottom_quintile.png',dpi=600)
    plt.clf()
 
    in_frontier = []
    for i in metrics_of_feasible.index:
        in_frontier.append(is_in_frontier(metrics_of_feasible,'total_costs', 'perc_spent_by_bottom_quintile', i))
    metrics_of_feasible.loc[:,'in_frontier'] = in_frontier
    
    sns.scatterplot(metrics_of_feasible, x='total_costs', y='perc_spent_by_bottom_quintile', hue='in_frontier')
    plt.title('Comparison 1')
    plt.xlabel('Total cost for the population/economy')
    plt.ylabel('Percentage of total water cost paid by bottom quintile')
    plt.savefig('output/scenarios/2b-perc_spent_by_bottom_quintile.png',dpi=600)
    plt.clf()

    sns.scatterplot(metrics, x='total_costs', y='per_gallon_price_bottom_quintile', hue='total_use')
    plt.xlabel('Total cost for the population/economy')
    plt.ylabel('Average water price for bottom quintile (in acre-feet/USD)')
    plt.savefig('output/scenarios/3-per_gallon_price_bottom_quintile.png',dpi=600)
    plt.clf()

    sns.scatterplot(metrics, x='total_costs', y='prop_bottom_used_compared_to_top', hue='total_use')
    plt.xlabel('Total cost for the population/economy')
    plt.ylabel('Proportion of water usage by bottom quintile compared to top quintile')
    plt.savefig('output/scenarios/4-prop_bottom_used_compared_to_top.png',dpi=600)
    plt.clf()
    
    sns.scatterplot(metrics, x='total_costs', y='prop_per_gallon_price', hue='total_use')
    plt.xlabel('Total cost for the population/economy')
    plt.ylabel('Proportion of avg. water price for bottom vs. top quintile')
    plt.savefig('output/scenarios/5-prop_per_gallon_price.png',dpi=600)
    plt.clf()

    print(" > Five plots saved to output/scenarios.")
    print(" > Done.")