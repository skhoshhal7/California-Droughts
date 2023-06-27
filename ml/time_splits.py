import pandas as pd
import yaml
import numpy as np


def convert_date_to_period(date):
    '''
    Converts date into period. Period 0 is January 2014
    '''
    datetime = pd.to_datetime(date)
    months_since_jan_2014 = (datetime.year - 2014) * 12 + datetime.month - 1
    return months_since_jan_2014

def determine_splits(config):
    print(" > Determining time splits...")
    time_config = config['temporal_config']

    feature_start = convert_date_to_period(time_config['feature_start_time'])
    feature_end = convert_date_to_period(time_config['feature_end_time'])
    label_start = convert_date_to_period(time_config['label_start_time'])
    label_end = convert_date_to_period(time_config['label_end_time'])


    split_df = pd.DataFrame(columns=[
        'feature_start',
        'feature_end',
        'train_label_start',
        'train_label_end',
        'test_feature_start',
        'test_feature_end',
        'test_label_start',
        'test_label_end',
        'group'])

    test_label_starts = np.arange(
        start=label_start,
        stop=label_end - time_config['test_label_timespans'],
        step=time_config['model_update_frequency'])

    j=0
    group=0

    # constructing several time chops as described in our report.
    for test_start in test_label_starts:
        train_label_ends = np.arange(
            test_start - time_config['max_training_histories'] + time_config['training_label_timespans'],
            test_start + 1,
            step = time_config['training_as_of_date_frequencies'])
        test_label_starts = train_label_ends - time_config['training_label_timespans']
        feature_ends = test_label_starts
        feature_starts = np.maximum(test_start - time_config['max_training_histories'], feature_start)
        feature_starts = np.minimum(feature_ends - time_config['min_training_histories'], feature_starts)

        test_feature_end = test_start
        test_feature_start = np.maximum(test_start - time_config['max_training_histories'], feature_start)
        test_feature_start = np.minimum(test_feature_end - time_config['min_training_histories'], test_feature_start)
        
        
        # storing splits in a dataframe
        for i in range(len(feature_ends)):
            new_split = {
                'feature_start': feature_starts[i],
                'feature_end': feature_ends[i],
                'train_label_start': test_label_starts[i],
                'train_label_end': train_label_ends[i],
                'test_feature_start': test_feature_start,
                'test_feature_end': test_feature_end,
                'test_label_start': test_start,
                'test_label_end': test_start + time_config['test_label_timespans'],
                'group': group
            }
            new_split = pd.DataFrame(new_split, index=[j])
            split_df = pd.concat([split_df,new_split], ignore_index=True)

            j+=1
        group += 1

    split_df = split_df[(split_df >= 0).all(1)]
    return split_df



def generating_splits(split_df, df, labels):
    print(" > Generating splits...")
    df = df.merge(labels, on=['fips', 'period'], how='left')

    X_trains = []
    y_trains = []
    X_tests = []
    y_tests = []
    groups = []

    # chopping off data at the previously defined time chops
    for group in split_df['group'].unique():
        for i in split_df.loc[split_df['group'] == group,:].index:
            
            X_trains.append(df.loc[
                (df['period']>=split_df.loc[i,'feature_start']) &
                (df['period']<split_df.loc[i,'feature_end']),
                :])
            y_trains.append(df.loc[df['period'] == split_df.loc[i,'feature_end'] - 1, ['fips','y']].drop_duplicates())

            X_tests.append(df.loc[
                (df['period']>=split_df.loc[i,'test_feature_start']) &
                (df['period']<split_df.loc[i,'test_feature_end']),
                :])
            y_tests.append(df.loc[df['period'] == split_df.loc[i,'test_feature_end'] - 1, ['fips','y']].drop_duplicates())
            groups.append(group)
    return X_trains, y_trains, X_tests, y_tests, groups


def split_data(features, labels):
    '''
    Generates time splits based on config.yaml file and splits data accordingly.
    '''
    config_path = 'ml/config.yaml'
    with open(config_path, 'r') as dbf:
        config = yaml.safe_load(dbf)

    split_df = determine_splits(config)
    validation_sets = generating_splits(split_df, features, labels)
    return validation_sets


def rbind_df(validation_sets_eng):
    print(" > Binding tables together...")
    X_trains, y_trains, X_tests, y_tests, groups = validation_sets_eng
    X_train_bind = [X_trains[0]]
    y_train_bind = [y_trains[0]]
    for i in range(1,len(X_trains)):
        if groups[i] == groups[i-1]:
            X_train_bind[-1] = pd.concat(
                [X_train_bind[-1],X_trains[i]],
                ignore_index=True)
            y_train_bind[-1] = pd.concat(
                [y_train_bind[-1],y_trains[i]],
                ignore_index=True)            
        else:
            X_train_bind.append(X_trains[i])
            y_train_bind.append(y_trains[i])

    X_test_bind = [X_tests[0]]
    y_test_bind = [y_tests[0]]
    for i in range(1,len(X_tests)):
        if groups[i] == groups[i-1]:

            X_test_bind[-1] = pd.concat(
                [X_test_bind[-1],X_tests[i]],
                ignore_index=True)
            y_test_bind[-1] = pd.concat(
                [y_test_bind[-1],y_tests[i]],
                ignore_index=True)      
        else:
            X_test_bind.append(X_tests[i])
            y_test_bind.append(y_tests[i])
            
    return X_train_bind, y_train_bind, X_test_bind, y_test_bind, groups
    