from ml.run_ml import run_ml
from sim.run_sim import run_sim

# 1: Runs machine learning pipeline to predict future water production
amount_available = run_ml()

# 2: Feeds predicted water production into our simulation
run_sim(amount_available)