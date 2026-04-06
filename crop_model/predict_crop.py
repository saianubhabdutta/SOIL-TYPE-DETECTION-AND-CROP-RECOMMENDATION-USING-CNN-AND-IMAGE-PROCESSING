import pickle
import numpy as np
import pandas as pd
from crop_model.feature_generator import get_features_from_soil

model = pickle.load(open("crop_model/crop_model.pkl", "rb"))
scaler = pickle.load(open("crop_model/scaler.pkl", "rb"))


columns = ['N','P','K','ph','EC','S','Cu','Fe','Mn','Zn','B']

def predict_from_soil(soil_type):

    features = get_features_from_soil(soil_type)

    # Convert to DataFrame (fix warning)
    df = pd.DataFrame([features], columns=columns)

    features_scaled = scaler.transform(df)

    probs = model.predict_proba(features_scaled)[0]

    top_indices = probs.argsort()[-3:][::-1]

    return model.classes_[top_indices]


# TEST
if __name__ == "__main__":

    soil = "Alluvial_Soil"

    crops = predict_from_soil(soil)

    print("Soil:", soil)
    print("Top Crops:", crops)
