# LIBRARIES
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import shapiro

from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import pickle

class Preprocessor:
    def __init__(self, filepath = "ObesityDataSet2.csv"):
        self.filepath = filepath
        self.data = None
        self.bin_encoder = {
            "Gender": {
                "Male": 0,
                "Female": 1
            },
            "family_history_with_overweight": {
                "no": 0,
                "yes": 1
            },
            "FAVC": {
                "no": 0,
                "yes": 1
            },
            "SMOKE": {
                "no": 0,
                "yes": 1
            },
            "SCC": {
                "no": 0,
                "yes": 1
            },
            "MTRANS": {
                "Walking": 1,
                "Not_Walking": 0
            }
        }
        self.ord_encoder = {
            "CAEC": {
                "no": 0,
                "Sometimes": 1,
                "Frequently": 2,
                "Always": 3
            },
            "CALC": {
                "no": 0,
                "Sometimes": 1,
                "Frequently": 2,
                "Always": 3
            }
        }
        self.target_encoder = {
            "NObeyesdad": {
                "Insufficient_Weight": 0,
                "Normal_Weight": 1,
                "Overweight_Level_I": 2,
                "Overweight_Level_II": 3,
                "Obesity_Type_I": 4,
                "Obesity_Type_II": 5,
                "Obesity_Type_III": 6
            }
        }
        self.catcols = [
            "Gender", 
            "family_history_with_overweight", 
            "FAVC", 
            "CAEC", 
            "SMOKE", 
            "SCC", 
            "CALC", 
            "MTRANS", 
            "NObeyesdad"]
        self.inconsistent = "Age"

    def read_data(self):
        self.data = pd.read_csv(self.filepath)

    def divide(self):
        self.numcols = [col for col in self.data.columns if col not in self.catcols]

    def handle_inconsistencies(self):
        self.data[self.inconsistent] = self.data[self.inconsistent].astype(str).str.extract(r'(\d+)').astype(int)
        
    def drop_duplicates(self):
        self.data = self.data.drop_duplicates().reset_index(drop = True)

    def impute_missing_values(self):
        for col in self.data.columns[self.data.isna().any()]:
            if col in self.catcols:
                self.data.loc[self.data[col].isna(), col] = self.data[col].dropna().mode().iloc[0]
            elif col in self.numcols:
                self.data.loc[self.data[col].isna(), col] = self.data[col].dropna().mean()

    def add_bmi(self):
        self.data["BMI"] = self.data["Weight"] / (self.data["Height"] ** 2)

    def map_mtrans(self):
        self.data["MTRANS"] = self.data["MTRANS"].map(lambda x: "Walking" if x == "Walking" else "Not_Walking")
        
    def encode(self):
        self.data = self.data.replace(self.bin_encoder)
        self.data = self.data.replace(self.ord_encoder)
        self.data = self.data.replace(self.target_encoder)

    def define_x_y(self):
        x = self.data.drop(columns = ["NObeyesdad"])
        y = self.data["NObeyesdad"]
        return x, y, self.numcols

class Modelling:
    def __init__(self, x, y, numcols, test_size = 0.3, n_estimators = 100, learning_rate = 0.9, max_depth = 10, eval_metric = "logloss", random_state = 11):
        self.x = x
        self.y = y
        self.numcols = numcols
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, 
                                                                                self.y, 
                                                                                test_size = test_size, 
                                                                                random_state = random_state)
        self.scaler = RobustScaler()
        self.model = XGBClassifier(n_estimators = n_estimators, 
                                   learning_rate = learning_rate, 
                                   max_depth = max_depth, 
                                   eval_metric = eval_metric)

    def scale(self):
        self.x_train[self.numcols] = self.scaler.fit_transform(self.x_train[self.numcols])
        self.x_test[self.numcols] = self.scaler.transform(self.x_test[self.numcols])

    def train(self):
        self.model.fit(self.x_train, 
                       self.y_train)

    def evaluate(self):
        self.y_pred = self.model.predict(self.x_test)
        print("Classification Report:")
        print(classification_report(self.y_test, 
                                    self.y_pred))
        cm = confusion_matrix(self.y_test, 
                              self.y_pred)
        print("Confusion Matrix:")
        print(pd.DataFrame(cm))

    def model_save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.model, 
                        f)

# ----------------------------------------------------
preprocessor = Preprocessor()
preprocessor.read_data()
preprocessor.divide()
preprocessor.handle_inconsistencies()
preprocessor.drop_duplicates()
preprocessor.impute_missing_values()
preprocessor.add_bmi()
preprocessor.map_mtrans()
preprocessor.encode()
x, y, numcols = preprocessor.define_x_y()

# ----------------------------------------------------
modelling = Modelling(x, y, numcols)
modelling.scale()
modelling.train()
modelling.evaluate()
modelling.model_save("Obesity Level Prediction Classification Model Deployment in Machine Learning with Python Pickle File.pkl")