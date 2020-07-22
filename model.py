# Importing Dependencies
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pickle

def imputeWithMode(df):
    """
    Going through each columns and checking the type is object
    if it is object, impute it with most frequent value
    """
    for col in df:
        if df[col].dtypes == 'object':
            df[col] = df[col].fillna(df[col].mode().iloc[0])

def main():
    credit_df = pd.read_csv('credit_approval_data.csv')

    # Replace "?" with NaN
    credit_df.replace('?', np.NaN, inplace = True)

    # Convert Age to numeric
    credit_df["Age"] = pd.to_numeric(credit_df["Age"])

    credit_df.fillna(credit_df.mean(), inplace=True)


    for col in credit_df:
        if credit_df[col].dtypes == 'object':
            credit_df[col] = credit_df[col].fillna(credit_df[col].mode().iloc[0])

    credit_df=credit_df.drop(["ZipCode"],axis=1)

    # Label Encoder
    LE = LabelEncoder()
    #Using label encoder to convert into numeric types
    for col in credit_df:
        if credit_df[col].dtypes=='object':
            credit_df[col]=LE.fit_transform(credit_df[col])

    selected_feat = ['PriorDefault','YearsEmployed','CreditScore','Debt','Income','Age', 'Approved']

    credit_df1 = credit_df[selected_feat]

    #convert to categorical data to dummy data

    X,y = credit_df1.iloc[:,credit_df1.columns != 'Approved'] , credit_df1["Approved"]

    # Spliting the data into training and testing sets
    X_train, X_test, y_train, Y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=123)

    # Scaling X_train and X_test
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaledX_train = scaler.fit_transform(X_train)
    rescaledX_test = scaler.transform(X_test)
    rf = RandomForestClassifier(n_estimators=500)
    rf.fit(rescaledX_train, y_train)
    y_pred = rf.predict(rescaledX_test)
    print(rf.score(rescaledX_test, Y_test))

    filename = 'model.pkl'

    pickle.dump(rf, open(filename, 'wb'))

    model = pickle.load(open(filename, 'rb'))

    pred = [1,0,0,0.835,1,40.92]

    print(pred)
    print(model.predict([pred]))


if __name__ == '__main__':
    main()