import pandas as pd
import numpy as np

def main():
    df = pd.read_csv('USA_Housing.csv')
    X = df.iloc[:,0:5].values
    Y = df.iloc[:,5].values

    from sklearn.preprocessing import StandardScaler
    trans = StandardScaler()
    X = trans.fit_transform(X)

    n = len(X[:,0])

    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()

    
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold

    kf = KFold(n_splits = 5)

    r2_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index],X[test_index]
        Y_train, Y_test = Y[train_index],Y[test_index]
        lin_reg.fit(X_train, Y_train)
        predict = lin_reg.predict(X_test)
        r2_scores.append(r2_score(predict,Y_test))
        
    print("r2 scores:",r2_scores)
    length = int(len(X)/5)
    folds = []
    fold = []
    for i in range(4):
        folds += [X[i*length:(i+1)*length]]
        fold += [Y[i*length:(i+1)*length]]
    folds += [X[4*length:len(X)]]
    fold += [Y[4*length:len(Y)]]

    print(fold)
    print(folds)


    for i in range(5):
        A = folds[i].T.dot(folds[i])
        B = np.linalg.inv(A)
        C = B.dot(folds[i].T)
        print('beta[',i,']=',C.dot(fold[i]))
    beta = C.dot(fold[i])

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(folds[4], fold[4], test_size=0.3, random_state=42)

    reg = LinearRegression().fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    y_predict=X_train.dot(beta)


if __name__ == '__main__':
    main()