# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

import gradio as gr

# Load dataset
df = pd.read_csv('traffic_accidents.csv')

# Drop missing values
df.dropna(inplace=True)

# Feature and target separation
X = df.drop('Accident_Severity', axis=1)
y = df['Accident_Severity']

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Preprocessing pipelines
numerical_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# Full pipeline with classifier
model = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Define prediction function for Gradio
def predict_accident_severity(**inputs):
    input_df = pd.DataFrame([inputs])
    prediction = model.predict(input_df)
    return prediction[0]

# Create Gradio input components
inputs = []
for col in X.columns:
    if col in numerical_cols:
        inputs.append(gr.Number(label=col))
    else:
        inputs.append(gr.Textbox(label=col))

# Launch Gradio interface
gr.Interface(
    fn=predict_accident_severity,
    inputs=inputs,
    outputs="text",
    title="Accident Severity Predictor"
).launch()
