import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset (replace with your dataset path)
data = pd.read_csv('KNN-APP.csv')

# Preprocess data (assuming you've already done this)
X = data.drop('target_column', axis=1)
y = data['target_column']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
kNN_model = KNeighborsClassifier(n_neighbors=7)
kNN_model.fit(X_train, y_train)

# Create the main window
root = tk.Tk()
root.title("KNN Prediction")

# Create input fields for features
feature_names = X.columns
input_vars = []
for i, feature in enumerate(feature_names):
    label = tk.Label(root, text=feature)
    label.pack()
    entry = tk.Entry(root)
    entry.pack()
    input_vars.append(entry)

# Create a predict button
def predict():
    # Get input values from entry fields
    input_data = [float(entry.get()) for entry in input_vars]
    # Reshape input data to match the model's expected format
    input_data = np.array(input_data).reshape(1, -1)
    # Make prediction
    prediction = kNN_model.predict(input_data)
    # Display the result
    result_label.config(text=f"Prediction: {prediction[0]}")

predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.pack()

# Create a label to display the result
result_label = tk.Label(root, text="")
result_label.pack()

# Create a frame to display sample data
sample_data_frame = tk.Frame(root)
sample_data_frame.pack()
# Add labels and text boxes to display sample data here (customize as needed)

# Start the GUI event loop
root.mainloop()