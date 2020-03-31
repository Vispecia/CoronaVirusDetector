import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

def data_split(data,ratio):
    np.random.seed(42)
    shuffle = np.random.permutation(len(data))
    test_size = int(len(data)*ratio)
    test_indices = shuffle[:test_size]
    train_indices  =shuffle[test_size:]
    return data.iloc[train_indices],data.iloc[test_indices]


if __name__ == "__main__":
    df = pd.read_csv('data.csv')
    train,test = data_split(df,0.4)
    x_train_features = train[['Fever','Age','Breath','RunnyNose','BodyPain']].to_numpy()
    x_train_labels = train[['Infection']].to_numpy().reshape(2100,)
    y_test_features = test[['Fever','Age','Breath','RunnyNose','BodyPain']].to_numpy()
    x_test_labels = test[['Infection']].to_numpy().reshape(1399,) 
    clf = LogisticRegression()
    clf.fit(x_train_features,x_train_labels)

    file = open('model.pkl','wb')
    pickle.dump(clf,file)
    file.close()

