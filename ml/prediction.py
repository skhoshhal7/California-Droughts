import yaml
from ml import features

def create_features(feature_df):
    '''
    Creates features over latest possible time split to predict water usage for the next 12 months
    '''
    config_path = 'ml/config.yaml'
    with open(config_path, 'r') as dbf:
        config = yaml.safe_load(dbf)
    time_config = config['temporal_config']
    history = time_config['max_training_histories']
    current_period = feature_df['period'].max()
    current_split = feature_df.loc[feature_df['period']<current_period,:]
    current_split = current_split.loc[current_split['period']>=current_period-history,:]

    X_trains = [current_split]
    X_tests = [current_split]
    y_trains = None
    y_tests = None
    groups = None
    datalist = X_trains, y_trains, X_tests, y_tests, groups

    data = features.generate_features(datalist, prediction=True)
    return data[0][0]

    