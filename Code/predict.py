from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import pickle

class Predict(object):

    def __init__(self):
        self.model=load_model('bbb_model')
        self.neuroinflammationmodel=pickle.load(open('polynomial_regression.sav','rb'))


    def preprocess_data(self,data):
        if type(data)==np.ndarray:
            data=np.float32(data)
            return self.normalize_data(data,False)
        elif type(data)==list():
            return self.normalize_data(np.array([data]),False)




    def normalize_data(self,data,flag=None):
        if flag:
            mean=np.load('mean.npy')
            std=np.load('std.npy')

            data=(data-mean)/std
            return data

        else:
            return data


    def predictlogBB(self,data):
        normalizeddata=self.preprocess_data(data)
        logBBresult=self.model.predict(normalizeddata.reshape(-1,16))
        print("Results predicted")
        return logBBresult[0][0]

    def neuroinflammationmodel(self,data):
        poly=PolynomialFeatures(degree=4,include_bias=False)
        poly_features=poly.fit_transform(data.values.reshape(-1,1))
        updated_logbb=self.neuroinflammationmodel.predict(poly_features)

        print("Neuro Inflammation model is run")

        return updated_logbb[0]



if __name__=="__main__":
    logBBpredictor=Predict()

    data=pd.read_csv('../Full_Dataset.csv')
    test=data.iloc[5,:-2]
    print(logBBpredictor.predictlogBB(test.values))
