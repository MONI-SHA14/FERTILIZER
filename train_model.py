import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# load dataset
data = pd.read_csv("fertilizer_data.csv")

# encode categorical data
le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fert = LabelEncoder()

data["Soil_Type"] = le_soil.fit_transform(data["Soil_Type"])
data["Crop_Type"] = le_crop.fit_transform(data["Crop_Type"])
data["Fertilizer"] = le_fert.fit_transform(data["Fertilizer"])

X = data.drop("Fertilizer", axis=1)
y = data["Fertilizer"]

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# save model
pickle.dump(model, open("../model.pkl", "wb"))

print("Model Trained Successfully")