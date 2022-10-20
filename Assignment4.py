from Linear_models_CK import GradientDescentWRRCK
import numpy as np

def main():
    X = np.random.rand(20,7)
    Y = np.random.rand(20,1)

    from sklearn.model_selection import train_test_split

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)
    
    lr = [0.0001,0.001,0.01,0.1,1,10]
    lambdas = [1e-15,1e-10,1e-5,1e-3,0,1,10,20]

    from sklearn.metrics import r2_score

    r2_scores = []

    minimum_r2 = -1000000
    best_coeff = np.array([])
    best_lr = 0
    best_rp = 0

    for lrates in lr:
        for rp in lambdas:
            model = GradientDescentWRRCK(lrates,50,rp)
            model.fit(X_train,Y_train)
            Y_pred = model.predict(X_test)
            r2_scores.append(r2_score(Y_pred,Y_test))
            if minimum_r2 < r2_score(Y_pred,Y_test):
                best_coff = model.B
                best_lr = lrates
                best_rp = rp
                minimum_r2 = r2_score(Y_pred,Y_test)

    print("Best Coefficiants:")
    print(best_coff)
    print("Best Learning rate:",best_lr)
    print("Best regularization parameter:",best_rp)

if __name__ == "__main__":
    main()
