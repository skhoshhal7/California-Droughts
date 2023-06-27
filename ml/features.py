import pandas as pd
import addfips
import numpy as np

def read_data():
    print(" > Reading in features...")
    # read in features and label data
    path = 'data/weather.csv'
    df = pd.read_csv(path)

    # create column that counts the months since January 2014
    df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['period'] = (df['year'] - 2014) * 12 + df['month'] - 1

    # rename fips column
    df.rename({'name': 'fips'}, axis=1, inplace=True)
    df.drop(['sealevelpressure', 'visibility'], axis=1, inplace=True)

    return df
    


def na_imputation(X_trains_eng, y_trains, X_tests_eng, y_tests):
    print(" > Imputing missing values...")
    X_trains_new = []
    y_trains_new = []
    X_tests_new = []
    y_tests_new = []
    for i in range(len(X_trains_eng)):
        X_trains_eng[i].reset_index(drop=True, inplace=True)
        y_trains[i].reset_index(drop=True, inplace=True)
        X_tests_eng[i].reset_index(drop=True, inplace=True)
        y_tests[i].reset_index(drop=True, inplace=True)

        X_trains_new.append(X_trains_eng[i].loc[(~y_trains[i]['y'].isna()),:])
        y_trains_new.append(y_trains[i].loc[~y_trains[i]['y'].isna(),:])
        X_tests_new.append(X_tests_eng[i].loc[~y_tests[i]['y'].isna(),:])
        y_tests_new.append(y_tests[i].loc[~y_tests[i]['y'].isna(),:])
    
    return X_trains_new, y_trains_new, X_tests_new, y_tests_new



def generate_features(datalist, prediction=False):
    print(" > Generating features...")
    X_trains, y_trains, X_tests, y_tests, groups = datalist

    X_trains_eng = []
    X_tests_eng = []
    for i in range(len(X_trains)):
        train = X_trains[i][['fips', 'latitude', 'longitude']].copy()
        test = X_tests[i][['fips', 'latitude', 'longitude']].copy()
        train.drop_duplicates(inplace=True)
        test.drop_duplicates(inplace=True)


        continuous = [
            'tempmax', 'tempmin', 'temp', 'feelslikemax',
            'feelslikemin', 'feelslike', 'dew', 'humidity', 'precip', 'precipprob',
            'precipcover', 'snow', 'snowdepth',
            'windspeed', 'winddir', 'cloudcover', 
            'solarradiation', 'solarenergy', 'uvindex' 
        ]
        categorical = [
            'conditions', 'icon', 'preciptype'
        ]
        imputations = ['snow', 'snowdepth']

        
        
        for j in ['train', 'test']:
            if j == 'train':
                source = X_trains[i]
                base = train
            elif j == 'test':
                source = X_tests[i]
                base = test
            else:
                raise ValueError

            ## imputing 0 for missing values
            source[imputations] = source[imputations].fillna(0)
            

            ## monthly avg
            avgs = source \
                .groupby(['fips', 'period'])[continuous] \
                .mean() \
                .reset_index()


            ## add monthly avg
            avgs_monthly = avgs \
                .groupby(['fips'])[continuous] \
                .mean() \
                .reset_index()
            train = base.merge(avgs_monthly, on=['fips'], how='left')
            for var in continuous:
                train.rename({var:'avg_'+var}, axis=1, inplace=True)


            ## add monthly max
            max_monthly = avgs \
                .groupby(['fips'])[continuous] \
                .max() \
                .reset_index()
            train = train.merge(max_monthly, on=['fips'], how='left')
            for var in continuous:
                train.rename({var:'max_'+var}, axis=1, inplace=True)


            ## add min
            min_monthly = avgs \
                .groupby(['fips'])[continuous] \
                .max() \
                .reset_index()
            train = train.merge(min_monthly, on=['fips'], how='left')
            for var in continuous:
                train.rename({var:'min_'+var}, axis=1, inplace=True)

            fips = pd.get_dummies(train.fips, prefix='fips')
            train = train.join(fips)

            if j == 'train':
                X_trains_eng.append(train)
            elif j == 'test':
                X_tests_eng.append(train)
            else:
                raise ValueError

    if not prediction:
        X_trains_eng, y_trains, X_tests_eng, y_tests = na_imputation(X_trains_eng, y_trains, X_tests_eng, y_tests)
            
    return X_trains_eng, y_trains, X_tests_eng, y_tests, groups
    


