# Water Price Optimization for the California Drought

This pipeline attempts to
(a) predict the amount of water available in a specific county in California for the next year and
(b) simulate different water pricing scenario, scoring them by economic impact and equitable cost distribution in society.


## Structure

Our code base is structured in the following way:

    .
    ├── data                    # contains the necessary data to run the pipeline
    ├── eda                     # The last training split is stored here for manual checking, if needed
    ├── ml                      
    |     ├── config.yaml       # Define the temporal parameters on how to split our data. Can be modified by user
    |     ├── features          # feature engineering is done here
    |     ├── labels            # reads in data for labels, and cleans and computes total annual per capita water production
    |     ├── prediction        # used to make predictions with the best model after training is complete
    |     ├── run_ml            # run file to run entire ml module
    |     ├── time_splits       # generates time splits based on config.yaml parameters
    |     └── training          # trains and validates models
    ├── output 
    |     ├── ml_performance    # stores model performance of every model on every time split
    |     └── scenarios         # stores output plots for policymakers to find best scenario
    ├── sim
    |     ├── econ              # makes economic calculations on predicted water usage and simulates average Californian town
    |     ├── plot              # plots results along three equity metrics
    |     ├── run_sim           # run file to run entire sim module
    |     └── simulation        # simulates water usage based on previously defined demand and supply
    ├── .gitignore              
    ├── requirements.txt        # all Python libraries needed for the project
    └── run.py                  # Run file to run entire pipeline. Only this file should be executed.




## How to run

**Clone the repository**

First, navigate to the directory where the repository should be located

Then clone the repository
```
git clone git@github.com:sebdodt/drought-policy-optimization.git
```


**Set up environment**

Create a new virtual environment
```
conda create --name venv python=3.10.6
conda activate venv
```

Install dependencies
```
pip install -r requirements.txt
```


**Run the pipeline**

You can run the pipeline with the following command
```
python run.py
```


**Results**

The results of the machine learning training will be stored in `output/ml_performance`.

The results from the simulation will be stored in `output/scenarios`.