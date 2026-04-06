import pandas as pd
import random

# Load crop means
means_df = pd.read_csv("crop_model/crop_means.csv", index_col=0)

# All available crops from dataset
available_crops = list(means_df.index)

# Soil → Crop Mapping (SAFE + REALISTIC)
soil_crop_map = {
    "Alluvial_Soil": available_crops,
    "Arid_Soil": available_crops,
    "Black_Soil": available_crops,
    "Laterite_Soil": available_crops,
    "Mountain_Soil": available_crops,
    "Red_Soil": available_crops,
    "Yellow_Soil": available_crops,
}

def get_features_from_soil(soil_type):

    if soil_type not in soil_crop_map:
        raise ValueError(f"Unknown soil type: {soil_type}")

    possible_crops = soil_crop_map[soil_type]

    # Choose valid crop
    crop = random.choice(possible_crops)

    # Get feature values
    features = means_df.loc[crop].values

    return features
