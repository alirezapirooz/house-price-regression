from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

import pandas as pd

df = pd.read_csv("cleaned_area.csv")

#df["Area"] = pd.to_numeric(df["Area"], errors="coerce")

#encode boolean features
bool_cols = ["Parking", "Warehouse", "Elevator"]

for col in bool_cols:
    df[col] = df[col].astype(int)
    
#featurs
X = df[["Area", "Room", "Parking", "Warehouse", "Elevator"]]
y = df["Price"]

#spilit
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
#train
model = LinearRegression()
model.fit(X_train, y_train)

#pred
y_pred = model.predict(X_test)

print("MAE: ", mean_absolute_error(y_test,y_pred))
print("R2_score: ", r2_score(y_test,y_pred))

sample = pd.DataFrame([{
    "Area": 100,
    "Room": 2,
    "Parking":1,
    "Warehouse":1,
    "Elevator":1
}])

print("Predicted price: ", model.predict(sample))
