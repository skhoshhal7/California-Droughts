from ml import labels, features, time_splits, training, prediction
import numpy as np
import pandas as pd

def run_ml():
    # define labels
    supply = labels.read_data()
    supply = labels.data_cleaning(supply)
    label_df = labels.label_construction(supply)
    label_df = labels.match_counties(label_df)

    # define features
    feature_df = features.read_data()
    validation_sets = time_splits.split_data(feature_df, label_df)
    validation_sets_eng = features.generate_features(validation_sets)
    validation_sets_bound = time_splits.rbind_df(validation_sets_eng)

    # model training
    rmse, model_names, models = training.train(validation_sets_bound)
    pd.DataFrame(rmse,columns=model_names).to_csv('output/ml_performance/ml_output.csv')

    # model selection
    best_model = model_names[np.argmin(rmse.mean(axis=0))]
    print(" > The best model is: ", best_model)
    print(" > The test RMSE error is: ", rmse.mean(axis=0)[np.argmin(rmse.mean(axis=0))])
    print(" > (Baseline RMSE is {base})".format(base=rmse.mean(axis=0)[0]))

    # model predictions for upcoming 12 months
    print(" > Using best model for predictions...")
    X_pred = prediction.create_features(feature_df)
    model = models[best_model]
    y_pred = model.predict(np.array(X_pred))

    # output final available water amount
    available_amount_for_avg_county = y_pred.mean()/125851 # divide by 125851 to convert from gallons to acre-feet for further calculation
    print(" > The average prediction is {avg} gallons water production per capita per year.".format(avg=y_pred.mean()))
    print(" > That is equivalent to {af} acre-feet of water.".format(af = available_amount_for_avg_county))

    print(" > Done.")
    return available_amount_for_avg_county 



if __name__=='__main__':
    run_ml()