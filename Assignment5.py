import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    ## Question 1

    from sklearn.datasets import load_iris

    iris = load_iris()

    X = iris.data[:, :4]  
    Y = iris.target

    from sklearn.model_selection import train_test_split

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

    from sklearn.linear_model import LogisticRegression

    logmod = LogisticRegression(multi_class='ovr', solver='liblinear')
    logmod.fit(X_train,Y_train)
    Y_pred = logmod.predict(X_test)
    print("Original Values: ",Y_test)
    print("Predicted Values:",Y_pred)

    from sklearn.metrics import r2_score

    logr2score = r2_score(Y_pred,Y_test)

    print("R2_score:",logr2score) 

    ## Question 2

    df = pd.read_csv("exam6.csv")

    X = df.iloc[:,:2].values
    Y = df.iloc[:,2].values

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

    # Points Plotting
    # plt.plot(X[:,0],X[:,1],'o')
    # plt.xlabel="Test1"
    # plt.ylabel="Test2"
    # plt.show()

    lr = LogisticRegression(max_iter=10,penalty='none')
    lr.fit(X_train,Y_train)
    Y_pred = lr.predict(X_test)
    print("LogisticRegression without Ridge regularization")
    print("Original Values: ",Y_test)
    print("Predicted Values:",Y_pred)
    print("R2_score:",r2_score(Y_pred,Y_test))

    lrwr = LogisticRegression(C=5,max_iter=1000,penalty='l2')
    lrwr.fit(X_train,Y_train)
    Y_pred = lrwr.predict(X_test)
    print("LogisticRegression with Ridge regularization")
    print("Original Values: ",Y_test)
    print("Predicted Values:",Y_pred)
    print("R2_score:",r2_score(Y_pred,Y_test))


if __name__ == "__main__":
    main()