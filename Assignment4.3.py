import numpy as np
import Linear_models_CK
from sklearn.datasets import load_boston
import warnings

with warnings.catch_warnings():
	warnings.filterwarnings("ignore")
	X, y = load_boston(return_X_y=True)

from sklearn.linear_model import RidgeCV,LassoCV

clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X, y)
print(clf.score(X, y))

reg = LassoCV(cv=5, random_state=0).fit(X, y)
print(reg.score(X, y))
reg.predict(X[:1,])