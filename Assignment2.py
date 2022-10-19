import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("AWCustomers.csv")
    dataset_sales = pd.read_csv("AWSales.csv")

    df = df[["CustomerID","CountryRegionName","BirthDate","Education","Occupation","Gender","MaritalStatus",
                       "NumberCarsOwned","NumberChildrenAtHome","TotalChildren","YearlyIncome"]]

    
    ohe_countries=pd.get_dummies(df['CountryRegionName'],drop_first=True)
    df.drop(columns=['CountryRegionName'],axis=1,inplace=True)
    df=pd.concat([ohe_countries,df],axis=1)

    df['BirthDate']= pd.to_datetime(df['BirthDate'])

    import datetime
    CURRENT_TIME = datetime.datetime.now()
    def get_age(birth_date,today=CURRENT_TIME):
        y=today-birth_date
        return y.days//365

    df['Age']=df['BirthDate'].apply(lambda x: get_age(x))

    df.drop(['BirthDate'],axis=1,inplace=True)


    df['Education']=df['Education'].map({'Partial High School':1,'High School':2,'Partial College':3,'Bachelors':4,'Graduate Degree':5})
    df['Occupation']=df['Occupation'].map({'Manual':1,'Skilled Manual':2,'Clerical':3,'Management':4,'Professional':5})

    df['Male']=df['Gender'].map({'M':1,'F':0})
    df.drop(['Gender'],axis=1,inplace=True)
    df['MaritalStatus']=df['MaritalStatus'].map({'M':1,'S':0})

    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler()
    scaled=scaler.fit_transform(df[['YearlyIncome','Age']])
    df['YearlyIncome_scaled']=scaled[:,0]
    df['Age_scaled']=scaled[:,1]
    df.drop(['YearlyIncome','Age'],axis=1,inplace=True)

    df = df.drop_duplicates(subset = ['CustomerID'])
    X = df.iloc[:,0:11].values
    Y = dataset_sales.iloc[:,1:3].values
    
    print(df)
     
    from scipy.spatial import distance
    print("Cosine Distance:",distance.cosine(df['Education'].values,df['YearlyIncome_scaled'].values))
    
    print("Jaccard Distance:",distance.jaccard(df['Education'].values,df['YearlyIncome_scaled'].values))
    
    from scipy.stats import pearsonr
    print("pearsonr:",pearsonr(df['Education'].values,df['YearlyIncome_scaled'].values)[0])

    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(X, Y)


if __name__ == "__main__":
    main()