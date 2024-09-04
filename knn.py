import tkinter as tk
from tkinter import ttk
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

root = tk.Tk()
root.title("KNN Body Type Predictor")
root.geometry("1000x600")  
root.configure(bg="#f5f5f5")  

def predict():
    try:
        sex = int(sex_entry.get())
        if sex not in [0, 1]:
            raise ValueError("Sex must be 0 (Female) or 1 (Male).")

        age = float(age_entry.get())
        weight = float(weight_entry.get())
        height = float(height_entry.get())

        if age <= 0 :
            raise ValueError("Age must be greater than 0.")
        if weight <= 0:
            raise ValueError("Weight must be greater than 0.")
        if height <= 0:
            raise ValueError("Height must be greater than 0.")

        new_data = np.array([[sex, age, weight, height]])
        prediction = knn.predict(new_data)[0]
        result_label.config(text=f" = {prediction}")
    except ValueError as e:
        result_label.config(text=str(e))

def reset_form():
    sex_entry.delete(0, tk.END)
    age_entry.delete(0, tk.END)
    weight_entry.delete(0, tk.END)
    height_entry.delete(0, tk.END)
    result_label.config(text="")

def populate_table():
    for row in data.itertuples(index=False):
        tree.insert("", "end", values=row)

# Load data
data = pd.read_csv('KNN_APP.csv')

X = data[['SEX', 'AGE', 'WEIGHT (kg)', 'HEIGHT (cm)']].values  
y = data['BODY'].values  

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

header_label = tk.Label(root, text="About Your Body", fg="white", bg="#4caf50", font=("Helvetica", 16, "bold"), pady=10)
header_label.pack(fill='x')

# Create input frame on the left
input_frame = ttk.Frame(root, padding="10 10 10 10", relief='ridge', borderwidth=2)
input_frame.pack(side='left', padx=20, pady=20, fill='y')

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

predict_button = ttk.Button(input_frame, text="Predict", command=predict)
predict_button.grid(row=4, column=0, columnspan=2, pady=10)

reset_button = ttk.Button(input_frame, text="Reset", command=reset_form)
reset_button.grid(row=5, column=0, columnspan=2, pady=10)

result_label = tk.Label(input_frame, text="", font=("Helvetica", 14), bg="#f5f5f5")
result_label.grid(row=6, column=0, columnspan=2, pady=10)

# Create a frame for the table on the right
table_frame = ttk.Frame(root, padding="10 10 10 10" ,borderwidth=2)
table_frame.pack(side='right', fill='both', expand=True, padx=20, pady=20)

# Create the Treeview for the table
tree = ttk.Treeview(table_frame, columns=list(data.columns), show="headings", height=20)
tree.pack(side='left', fill='both', expand=True)

# Scrollbars
vsb = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
vsb.pack(side='right', fill='y')
tree.configure(yscrollcommand=vsb.set)

hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
hsb.pack(side='bottom', fill='x')
tree.configure(xscrollcommand=hsb.set)

# Define the column headings
for col in data.columns:
    tree.heading(col, text=col)
    tree.column(col, width=100, anchor='w')

populate_table()

root.mainloop()
