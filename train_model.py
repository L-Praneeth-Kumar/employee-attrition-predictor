import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os

# Load the dataset
df = pd.read_csv('Employee-Attrition.csv')

# Drop irrelevant columns
# EmployeeCount, StandardHours, Over18 are constants
# EmployeeNumber is an ID
df = df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1)

# Separate features and target
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Encode target variable
le = LabelEncoder()
y = le.fit_transform(y) # No: 0, Yes: 1

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Fit and transform the features
X_processed = preprocessor.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Build the ANN
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=32, activation='relu', input_dim=X_train.shape[1]),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
print("Training the model...")
model.fit(X_train, y_train, batch_size=32, epochs=50, validation_split=0.1, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Create assets directory if it doesn't exist
os.makedirs('assets', exist_ok=True)

# Save the model and preprocessor
model.save('assets/model.keras')
joblib.dump(preprocessor, 'assets/preprocessor.joblib')
joblib.dump(categorical_cols, 'assets/categorical_cols.joblib')
joblib.dump(numerical_cols, 'assets/numerical_cols.joblib')

# Save class mapping for reference in app
joblib.dump(le.classes_, 'assets/label_encoder_classes.joblib')

print("Model and preprocessing assets saved successfully in 'assets/' folder.")
