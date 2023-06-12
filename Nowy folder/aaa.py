import tkinter as tk

class MyApp:
    def __init__(self):
        self.window = tk.Tk()
        self.create_tab1_widgets()

    def create_tab1_widgets(self):
        self.script1_button = tk.Button(self.window)
        self.script1_button["text"] = "Reguły asocjacyjne - wartości domyślne"
        self.script1_button["command"] = self.add_text
        self.script1_button.pack(side=tk.TOP)

        self.text_field = tk.Text(self.window, height=10, width=30)
        self.text_field.pack()

    def add_text(self):
        self.text_field.insert(tk.END, """
import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

st.header("Analiza reguł asocjacyjnych")

uploaded_file = st.file_uploader("Choose a file")  # Uploader plików

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)  # Odczytanie zbioru danych
    st.write(df)  # Wypisanie data frame pliku

    min_support = st.slider('Minimum support', min_value=0.0, max_value=1.0, value=0.1, step=0.01)  # Minimalne wsparcie
    min_confidence = st.slider('Minimum confidence', min_value=0.0, max_value=1.0, value=0.5, step=0.01)  # Minimalne zaufanie

    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

    # Filtruj reguły według minimalnego zaufania
    rules = rules[rules['confidence'] >= min_confidence]

    st.write("Liczba reguł znalezionych: ", len(rules))

    st.write("Najważniejsze reguły:")
    st.table(rules)
        """)

    def run(self):
        self.window.mainloop()

app = MyApp()
app.run()