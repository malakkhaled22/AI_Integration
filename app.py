from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("model(1).pkl")
features = joblib.load("features(1).pkl")
sub_area_map = joblib.load("sub_area_freq_map.pkl")

def norm(x):
    return str(x).strip().lower()

sub_area_map = {norm(k): float(v) for k, v in sub_area_map.items()}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame(0.0, index=[0], columns=features)

    size = float(data.get("Size_sqm", 0))
    bedrooms = float(data.get("Bedroom_Num", 0))
    bathrooms = float(data.get("bathrooms_numeric", 0))

    df.at[0, "Size_sqm"] = size
    df.at[0, "Bedroom_Num"] = bedrooms
    df.at[0, "bathrooms_numeric"] = bathrooms
    df.at[0, "is_land"] = float(data.get("is_land", 0))
    df.at[0, "is_cash"] = float(data.get("is_cash", 0))
    df.at[0, "has_bedrooms"] = 1.0 if bedrooms > 0 else 0.0
    df.at[0, "has_bathrooms"] = 1.0 if bathrooms > 0 else 0.0
    
    if size < 100:
        df.at[0, "Size_Category_Small"] = 1.0
    elif 100 <= size <= 200:
        df.at[0, "Size_Category_Medium"] = 1.0

    if size > 250 and bedrooms >= 4:
        df.at[0, "Luxury_Property"] = 1.0

    main_area = data.get("main_area", "")
    property_type = data.get("type", "")

    if f"main_area_{main_area}" in df.columns:
        df.at[0, f"main_area_{main_area}"] = 1.0
    
    if f"type_{property_type}" in df.columns:
        df.at[0, f"type_{property_type}"] = 1.0

    # 6. تردد المنطقة (Sub Area Frequency)
    sub_area = norm(data.get("sub_area", ""))
    df.at[0, "sub_area_freq"] = sub_area_map.get(sub_area, 0.0)

    # ---------------- Debug ----------------
    active_features = df.loc[0, df.columns[df.iloc[0] != 0]]
    print("--- ACTIVE FEATURES SENT TO MODEL ---")
    print(active_features)
    # ---------------------------------------

    # 7. التنبؤ بالسعر
    prediction = float(model.predict(df)[0])
    
    return {"predicted_price": prediction}