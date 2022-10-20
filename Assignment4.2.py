from Linear_models_CK import GradientDescentWRRCK
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge,Lasso 

def main():
    df = pd.read_csv("Hitters.csv")
    rp = 0.5748

    meanVal = df['Salary'].mean()
    df['Salary'].fillna(value=meanVal, inplace=True)

    df['Division=E']=df['Division'].map({'E':1,'W':0})
    df.drop(['Division'],axis=1,inplace=True)

    df['League=A']=df['League'].map({'A':1,'N':0})
    df.drop(['League'],axis=1,inplace=True)

    #print(df)
    X = df.iloc[:,0:17].values
    Y = df.iloc[:,16].values

    from sklearn.model_selection import train_test_split
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

    from sklearn.metrics import r2_score

    
    ridge_mod = Ridge(alpha = rp)
    ridge_mod.fit(X_train,Y_train)
    Y_pred = ridge_mod.predict(X_test)
    ridge_r2 = r2_score(Y_pred,Y_test)
    print("Ridge r2_score:",ridge_r2)

    lasso_mod = Lasso(alpha = rp)
    lasso_mod.fit(X_train,Y_train)
    Y_pred = lasso_mod.predict(X_test)
    lasso_r2 = r2_score(Y_pred,Y_test)
    print("Lasso r2_score:",lasso_r2)

    if ridge_r2 >= lasso_r2:
        print("Ridge regression worked better")
    else:
        print("Lasso regression worked better")

if __name__ == "__main__":
    main()