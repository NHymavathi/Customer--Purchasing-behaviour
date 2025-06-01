import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
file_path = 'C:/Users/nhymavathi40gmail.c/OneDrive/Desktop/customer purchasing behaviour/Customer Purchasing Behaviors.csv'

df = pd.read_csv(file_path)

# Drop user_id if present
if 'user_id' in df.columns:
    df.drop('user_id', axis=1, inplace=True)

# Drop or fill missing values
df.dropna(inplace=True)

# Create a new binary target column 'will_buy'
# Example: customer likely to buy if income > 50K and frequency > 15
df['will_buy'] = np.where((df['annual_income'] > 50000) & (df['purchase_frequency'] > 15), 1, 0)

# Prepare features and target
X = df[['age', 'annual_income', 'purchase_amount', 'loyalty_score', 'purchase_frequency']]  # drop region
y = df['will_buy']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train, y_train)
joblib.dump(model, 'model.pkl')

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("✅ Model and scaler saved successfully.")
print(f"🎯 Model Accuracy: {accuracy * 100:.2f}%")
                                                        