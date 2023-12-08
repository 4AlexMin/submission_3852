from sota.sotas import get_sota_models
from model import SocialBucks
import numpy as np

# Acquire benchmarks
datasets =  (
    np.random.random(200).reshape(100,-1),
    np.random.randint(low=0, high=5, size=100)
    )    #...(Your dataset to be tested)
X, y = datasets
sota_name = 'smknn'
sota_model = get_sota_models()[sota_name]


# Fit and predict --  Directly perform any algorithm as a sklearn classifier (i.e., fit() and predict())
danng = SocialBucks()
danng.fit(X, y)
danng.predict(X)
sota_model.fit(X, y)
sota_model.predict(X)
