import pandas as pd
import numpy as np

def main():
    df = pd.read_csv('USA_Housing.csv')
    X = df.iloc[:,0:5].values
    Y = df.iloc[:,5].values

    from sklearn.preprocessing import StandardScaler
    trans = StandardScaler()
    X = trans.fit_transform(X)

    from sklearn.model_selection import train_test_split

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.44)
    X_val,X_test,Y_val,Y_test = train_test_split(X_test,Y_test,test_size=0.68181)

    lr = [0.001,0.01,0.1,1]

    from sklearn.linear_model import LinearRegression,SGDClassifier

    SGDClf = SGDClassifier(max_iter = 1000, tol=1e-3)
    SGDClf.fit(X_train,Y_train)


if __name__ == '__main__':
    main()