import tkinter as tk
from tkinter import filedialog
import pandas as pd
from pandastable import Table

def open_csv():
    filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if filepath:
        df = pd.read_csv(filepath)
        display_dataframe(df)

def display_dataframe(df):
    top = tk.Toplevel(root)
    top.title("DataFrame View")

    frame = tk.Frame(top)
    frame.pack(fill="both", expand=True)

    table = Table(frame, dataframe=df)
    table.show()

root = tk.Tk()
root.title("CSV Viewer")

button = tk.Button(root, text="Open CSV", command=open_csv)
button.pack(pady=10)

root.mainloop()