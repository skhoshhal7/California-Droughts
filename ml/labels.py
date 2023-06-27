import pandas as pd
import addfips

def read_data():
    print(" > Reading in data for labels...")
    # read in features and label data
    supply_path = 'data/supply.xlsx'
    supply = pd.read_excel(supply_path)
    return supply


def data_cleaning(supply):
    print(" > Cleaning label data...")
    # create column that counts the months since January 2014
    supply['month'] = supply['Reporting Month'].dt.month
    supply['year'] = supply['Reporting Month'].dt.year
    supply['period'] = (supply['year'] - 2014) * 12 + supply['month'] - 1
    supply.rename({'CALCULATED Total Potable Water Production Gallons (Ag Excluded)': 'production'}, 
        axis=1, 
        inplace=True)

    # some rows contain two counties. We split the water production equally between all counties
    supply['n_counties_served'] = supply['County'] \
        .str.count(',') \
        .reset_index(drop=True) + 1
    supply = supply \
        .set_index(supply.columns.drop('County',1).tolist()) \
        ['County'].str.split(',', expand=True) \
        .stack() \
        .reset_index() \
        .rename(columns={0:'County'}) \
        .loc[:, supply.columns]
    supply['production'] = supply['production'] / supply['n_counties_served']

    return supply


def label_construction(supply):
    '''
    Function that constructs the labels based on our dataset.

    Our label is the sum of water production in the 12 months following the current month.
    (The current month is not included in the sum.)
    '''
    print(" > Constructing labels...")
    # There are multiple producers for water in each county.
    # We will sum them up per month and per county.
    supply_sum = supply \
        .groupby(['County', 'period'])[['production','Total Population Served']] \
        .sum() \
        .reset_index()

    # We are taking the rolling sum of the last 12 months up to and including the current month
    supply.sort_values('period', axis=0, inplace=True)
    supply_sum['sum_current_year'] = supply_sum.groupby('County')['production'].rolling(12).sum().reset_index(0,drop=True)

    # We are shifting the data to get the sum of production for the upcoming 12 months excluding the current month
    supply_sum['sum_next_year'] = supply_sum.groupby(['County'])['sum_current_year'].shift(-12)
    
    # # We calculate the percentage change and call it `y`
    # supply_sum['y'] = (supply_sum['sum_next_year'] - supply_sum['sum_current_year']) / supply_sum['sum_current_year']

    # We calculate how many people are served
    supply_sum['served_current_year'] = supply_sum.groupby('County')['Total Population Served'].rolling(12).mean().reset_index(0,drop=True)
    supply_sum['served_next_year'] = supply_sum.groupby(['County'])['served_current_year'].shift(-12)

    # We calculate the per capita production
    supply_sum['y'] = supply_sum['sum_next_year'] / supply_sum['served_next_year']
    
    # Drop next year sum as it isn't needed
    supply_sum.drop(
        'sum_next_year',
        axis=1,
        inplace=True)

    return supply_sum


def match_counties(df):
    '''
    This function matches the county name to the FIPS code.
    '''
    print(" > Match counties to FIPS codes...")
    # initialise values
    state = "CA"
    af = addfips.AddFIPS()
    df['fips'] = None
    df.reset_index(drop=True, inplace=True)

    # loop through all counties to append fips codes
    for i in range(len(df)):
        df.loc[i,'fips'] = af.get_county_fips(df.loc[i,'County'], state=state)
    df['fips'] = df['fips'].astype(int)
    # All 50 counties are matched, California has 58 counties
    # Missing 8 are likely small counties without water production
    return df
    

if __name__=='__main__':
    supply = read_data()
    supply = data_cleaning(supply)
    labels = label_construction(supply)
    labels = match_counties(labels)
