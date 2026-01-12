import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 1. Load Data
# Make sure 'cardio_train.csv' is in the same folder!
df = pd.read_csv("cardio_train.csv", sep=";")

# 2. Clean Data (Drop ID as it is noise)
X = df.drop(["cardio", "id"], axis=1)
y = df["cardio"]

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scale (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train Model (Using Random Forest for better accuracy than Logistic Regression)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Save Model and Scaler
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("SUCCESS: model.pkl and scaler.pkl have been created!")