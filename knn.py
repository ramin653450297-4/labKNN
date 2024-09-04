import tkinter as tk
from tkinter import ttk
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

root = tk.Tk()
root.title("KNN Body Type Predictor")
root.geometry("400x400")
root.configure(bg="#f5f5f5")  

def predict():
    try:
        sex = float(sex_entry.get())
        age = float(age_entry.get())
        weight = float(weight_entry.get())
        height = float(height_entry.get())

        new_data = np.array([[sex, age, weight, height]])
        prediction = knn.predict(new_data)[0]
        result_label.config(text=f"Prediction: {prediction}")
    except ValueError:
        result_label.config(text="Invalid input!")

def reset_form():
    sex_entry.delete(0, tk.END)
    age_entry.delete(0, tk.END)
    weight_entry.delete(0, tk.END)
    height_entry.delete(0, tk.END)
    result_label.config(text="Prediction: ")

data = pd.read_csv('KNN_APP.csv')

X = data[['SEX', 'AGE', 'WEIGHT (kg)', 'HEIGHT (cm)']].values  
y = data['BODY'].values  


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)


header_label = tk.Label(root, text="About Your Body", fg="white", bg="#4caf50", font=("Helvetica", 16, "bold"), pady=10)
header_label.pack(fill='x')


input_frame = ttk.Frame(root, padding="10 10 10 10", relief='ridge', borderwidth=2)
input_frame.pack(padx=20, pady=20, fill='x', expand=True)


ttk.Label(input_frame, text="SEX (0=Female, 1=Male):").grid(row=0, column=0, sticky='w', pady=5)
sex_entry = ttk.Entry(input_frame)
sex_entry.grid(row=0, column=1, pady=5)

ttk.Label(input_frame, text="AGE:").grid(row=1, column=0, sticky='w', pady=5)
age_entry = ttk.Entry(input_frame)
age_entry.grid(row=1, column=1, pady=5)

ttk.Label(input_frame, text="WEIGHT (kg):").grid(row=2, column=0, sticky='w', pady=5)
weight_entry = ttk.Entry(input_frame)
weight_entry.grid(row=2, column=1, pady=5)

ttk.Label(input_frame, text="HEIGHT (cm):").grid(row=3, column=0, sticky='w', pady=5)
height_entry = ttk.Entry(input_frame)
height_entry.grid(row=3, column=1, pady=5)


predict_button = ttk.Button(root, text="Predict", command=predict)
predict_button.pack(pady=10)


reset_button = ttk.Button(root, text="Reset", command=reset_form)
reset_button.pack(pady=10)

result_label = tk.Label(root, text="Prediction: ", font=("Helvetica", 14), bg="#f5f5f5")
result_label.pack(pady=10)

root.mainloop()