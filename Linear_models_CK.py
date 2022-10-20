"""
Linear models library created by Chitraksh Kumar cus already present libs werent cutting it fr me
"""

import numpy as np

class LinearRegressionCK:
    def __init__(self):
        self.X = np.array([])
        self.y = np.array([])
        self.B = np.array([]) 

    def fit(self,X,Y):
        col = np.array([1 for i in range(len(X[:,0]))])
        X = np.insert(X,0,col,1) 
        self.B = np.matmul( np.linalg.inv(np.matmul(X.transpose(),X)) , np.matmul(X.transpose(),Y) )

    def predict(self,X):
        col = np.array([1 for i in range(len(X[:,0]))])
        X = np.insert(X,0,col,1) 
        return np.matmul(X,self.B)

class GradientDescentCK:
    def __init__(self,lrate=0.01,maxiter=1000):
        self.lrate = lrate
        self.maxiter = maxiter

    def fit(self,X,Y):
        self.n = len(X[:,0])
        self.k = len(X[0,:]) + 1
        col = np.array([1 for i in range(self.n)])
        X = np.insert(X,0,col,1)
        self.B = np.array([[0.0] for i in range(self.k)])
        for _ in range(self.maxiter):
            tB = np.array(self.B)
            for i in range(self.k):
                self.B[i] = tB[i] - (self.lrate/self.n)*(np.sum(np.matmul(np.matmul(X[:,i],X),tB)) - np.sum(np.matmul(X[:,i],Y)))

    def predict(self,X):
        col = np.array([1 for i in range(len(X[:,0]))])
        X = np.insert(X,0,col,1) 
        return np.matmul(X,self.B)


class GradientDescentWRRCK:
    def __init__(self,lrate=0.01,maxiter=1000,l=0):
        self.lrate = lrate
        self.maxiter = maxiter
        self.l = l

    def fit(self,X,Y):
        self.n = len(X[:,0])
        self.k = len(X[0,:]) + 1
        col = np.array([1 for i in range(self.n)])
        X = np.insert(X,0,col,1)
        self.B = np.array([[0.0] for i in range(self.k)])
        for _ in range(self.maxiter):
            tB = np.array(self.B)
            for i in range(self.k):
                self.B[i] = tB[i] - (self.lrate/self.n)*(np.sum(np.matmul(np.matmul(X[:,i],X),tB)) 
                                  - np.sum(np.matmul(X[:,i],Y)) + np.sum(self.l*tB[i]))

    def predict(self,X):
        col = np.array([1 for i in range(len(X[:,0]))])
        X = np.insert(X,0,col,1) 
        return np.matmul(X,self.B)


def main():
    # Multiple Linear Regression example 
    X = np.array([[7,560],[3,220],[3,340]])
    Y = np.array([[16.68],[11.50],[12.03]])
    X_test = np.array([[4,80],[5,470]])
    lin_mod = LinearRegressionCK()
    lin_mod.fit(X,Y)
    Y_test = lin_mod.predict(X_test)
    print(Y_test)

    # Gradient Descent example
    X = np.array([[2.75,5.3],[2.5,5.3],[2.25,5.5],[2,5.7],[2,5.9],[2,6],[1.75,5.9],[1.75,6.1]])
    Y = np.array([[1464],[1394],[1159],[1130],[1075],[1047],[965],[719]])
    X_test = np.array([[2.75,5.3],[2.5,5.3]])
    grad_mod = GradientDescentCK(lrate=0.01,maxiter=1000)
    grad_mod.fit(X,Y)
    Y_test = grad_mod.predict(X_test)
    print(Y_test)

    # Gradient Descent with ridge regression example
    X = np.array([[2.75,5.3],[2.5,5.3],[2.25,5.5],[2,5.7],[2,5.9],[2,6],[1.75,5.9],[1.75,6.1]])
    Y = np.array([[1464],[1394],[1159],[1130],[1075],[1047],[965],[719]])
    X_test = np.array([[2.75,5.3],[2.5,5.3]])
    gradwrr_mod = GradientDescentWRRCK(lrate=0.01,maxiter=1000,l=1)
    gradwrr_mod.fit(X,Y)
    Y_test = gradwrr_mod.predict(X_test)
    print(Y_test)

    from sklearn.model_selection import train_test_split
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv( "salary_data.csv" )
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, 1].values    
  
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y,     
                                          test_size = 1 / 3, random_state = 0 )

    model = GradientDescentWRRCK( lrate = 0.01,maxiter = 1000,l = 1)
    model.fit( X_train, Y_train )
    print(model.B)
    Y_pred = model.predict(X_test)
    print( "Predicted values ", np.round( Y_pred[:3], 2 ) )     
    print( "Real values      ", Y_test[:3] )  

    plt.scatter( X_test, Y_test, color = 'blue' )    
    plt.plot( X_test, Y_pred, color = 'orange' )    
    plt.title( 'Salary vs Experience' )    
    plt.xlabel( 'Years of Experience' )    
    plt.ylabel( 'Salary' )    
    plt.show()


if __name__ == "__main__":
    main()