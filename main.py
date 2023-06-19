import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
import subprocess

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Generator skryptów Streamlit")
        self.pack()
        self.create_widgets()

#Dodanie widżetów do GUI
    def create_widgets(self):
        self.tabControl = ttk.Notebook(self)
        self.tabControl.pack(fill="both", expand=True)

        self.tab1 = ttk.Frame(self.tabControl)
        self.tabControl.add(self.tab1, text="Reguły asocjacyjne")
        self.create_tab1_widgets()

        self.tab2 = ttk.Frame(self.tabControl)
        self.tabControl.add(self.tab2, text="Klasyfikator-KNN")
        self.create_tab2_widgets()

        self.tab4 = ttk.Frame(self.tabControl)
        self.tabControl.add(self.tab4, text="Klasyfikator-Drzewo losowe")
        self.create_tab4_widgets()

        self.tab5 = ttk.Frame(self.tabControl)
        self.tabControl.add(self.tab5, text="Klasyfikator-Drzewo decyzyjne")
        self.create_tab5_widgets()

        self.tab3 = ttk.Frame(self.tabControl)
        self.tabControl.add(self.tab3, text="Grupowanie-KMeans")
        self.create_tab3_widgets()

        self.tab6 = ttk.Frame(self.tabControl)
        self.tabControl.add(self.tab6, text="Grupowanie-DBSCAN")
        self.create_tab6_widgets()

        self.tab7 = ttk.Frame(self.tabControl)
        self.tabControl.add(self.tab7, text="Grupowanie-Hierarchiczna")
        self.create_tab7_widgets()

        self.script_text = tk.Text(self, height=20, width=100)
        self.script_text.pack()

        self.save_button = tk.Button(self, text="Zapisz", command=self.save_script)
        self.save_button.pack(side=tk.TOP)

        self.run_button = tk.Button(self, text="Uruchom", command=self.run_script)
        self.run_button.pack(side=tk.TOP)

        self.clear_button = tk.Button(self, text="Wyczyść", command=self.clear_script)
        self.clear_button.pack(side=tk.TOP)

    def create_tab1_widgets(self):
        self.script1_button = tk.Button(self.tab1)
        self.script1_button["text"] = "Reguły asocjacyjne - wartości domyślne"
        self.script1_button["command"] = self.add_text_ad
        self.script1_button.pack(side=tk.TOP)

        self.tab1_var = tk.IntVar()
        self.tab1_checkbox = tk.Checkbutton(self.tab1, text="Funkcje zaawansowane", variable=self.tab1_var,
                                            command=self.show_additional_functions)
        self.tab1_checkbox.pack(side=tk.TOP)

        self.tab1_additional_label1 = tk.Label(self.tab1, text="Wsparcie")
        self.tab1_additional_label1.pack(side=tk.LEFT)
        self.tab1_additional_label1.configure(state='disabled')

        self.tab1_additional_field1 = tk.Entry(self.tab1)
        self.tab1_additional_field1.pack(side=tk.LEFT)
        self.tab1_additional_field1.bind("<Return>", lambda event: self.add_additional_values_a())
        self.tab1_additional_field1.configure(state='disabled')

        self.tab1_additional_label2 = tk.Label(self.tab1, text="Zaufanie")
        self.tab1_additional_label2.pack(side=tk.LEFT)
        self.tab1_additional_label2.configure(state='disabled')

        self.tab1_additional_field2 = tk.Entry(self.tab1)
        self.tab1_additional_field2.pack(side=tk.LEFT)
        self.tab1_additional_field2.bind("<Return>", lambda event: self.add_additional_values_a())
        self.tab1_additional_field2.configure(state='disabled')

    def create_tab2_widgets(self):
        self.script2_button = tk.Button(self.tab2)
        self.script2_button["text"] = "Train and Test - wartości domyślne"
        self.script2_button["command"] = self.add_text_knn
        self.script2_button.pack(side=tk.TOP)

        self.tab2_var = tk.IntVar()
        self.tab2_checkbox = tk.Checkbutton(self.tab2, text="Funkcje zaawansowane", variable=self.tab2_var,
                                            command=self.show_additional_functions)
        self.tab2_checkbox.pack(side=tk.TOP)

        self.tab2_additional_label1 = tk.Label(self.tab2, text="Rozmiar tablicy testowej")
        self.tab2_additional_label1.pack(side=tk.LEFT)
        self.tab2_additional_label1.configure(state='disabled')

        self.tab2_additional_field1 = tk.Entry(self.tab2)
        self.tab2_additional_field1.pack(side=tk.LEFT)
        self.tab2_additional_field1.bind("<Return>", lambda event: self.add_additional_values_knn())
        self.tab2_additional_field1.configure(state='disabled')

        self.tab2_additional_label2 = tk.Label(self.tab2, text="Stan losowy")
        self.tab2_additional_label2.pack(side=tk.LEFT)
        self.tab2_additional_label2.configure(state='disabled')

        self.tab2_additional_field2 = tk.Entry(self.tab2)
        self.tab2_additional_field2.pack(side=tk.LEFT)
        self.tab2_additional_field2.bind("<Return>", lambda event: self.add_additional_values_knn())
        self.tab2_additional_field2.configure(state='disabled')

        self.tab2_additional_label3 = tk.Label(self.tab2, text="Liczba sąsiadów")
        self.tab2_additional_label3.pack(side=tk.LEFT)
        self.tab2_additional_label3.configure(state='disabled')

        self.tab2_additional_field3 = tk.Entry(self.tab2)
        self.tab2_additional_field3.pack(side=tk.LEFT)
        self.tab2_additional_field3.bind("<Return>", lambda event: self.add_additional_values_knn())
        self.tab2_additional_field3.configure(state='disabled')

        self.tab2_menu_label = tk.Label(self.tab2, text="Wybierz metrykę:")
        self.tab2_menu_label.pack(side=tk.LEFT)
        self.tab2_menu_label.configure(state='disabled')

        self.tab2_menu = ttk.Combobox(self.tab2, values=["euclidean", "manhattan", "minkowski"])
        self.tab2_menu.pack(side=tk.LEFT)
        self.tab2_menu.bind("<Return>", lambda event: self.add_additional_values_knn())
        self.tab2_menu.configure(state='disabled')

    def create_tab4_widgets(self):
        self.script4_button = tk.Button(self.tab4)
        self.script4_button["text"] = "Train and Test - wartości domyślne"
        self.script4_button["command"] = self.add_text_dl
        self.script4_button.pack(side=tk.TOP)
        self.tab4_var = tk.IntVar()
        self.tab4_checkbox = tk.Checkbutton(self.tab4, text="Funkcje zaawansowane", variable=self.tab4_var,
                                            command=self.show_additional_functions)
        self.tab4_checkbox.pack(side=tk.TOP)
        self.tab4_additional_label1 = tk.Label(self.tab4, text="Rozmiar tablicy testowej")
        self.tab4_additional_label1.pack(side=tk.LEFT)
        self.tab4_additional_label1.configure(state='disabled')

        self.tab4_additional_field1 = tk.Entry(self.tab4)
        self.tab4_additional_field1.pack(side=tk.LEFT)
        self.tab4_additional_field1.bind("<Return>", lambda event: self.add_additional_values_dl())
        self.tab4_additional_field1.configure(state='disabled')

        self.tab4_additional_label2 = tk.Label(self.tab4, text="Stan losowy")
        self.tab4_additional_label2.pack(side=tk.LEFT)
        self.tab4_additional_label2.configure(state='disabled')

        self.tab4_additional_field2 = tk.Entry(self.tab4)
        self.tab4_additional_field2.pack(side=tk.LEFT)
        self.tab4_additional_field2.bind("<Return>", lambda event: self.add_additional_values_dl())
        self.tab4_additional_field2.configure(state='disabled')

        self.tab4_additional_label3 = tk.Label(self.tab4, text="Liczba estymatorów")
        self.tab4_additional_label3.pack(side=tk.LEFT)
        self.tab4_additional_label3.configure(state='disabled')

        self.tab4_additional_field3 = tk.Entry(self.tab4)
        self.tab4_additional_field3.pack(side=tk.LEFT)
        self.tab4_additional_field3.bind("<Return>", lambda event: self.add_additional_values_dl())
        self.tab4_additional_field3.configure(state='disabled')

        self.tab4_additional_label4 = tk.Label(self.tab4, text="Maksymalna głębokość")
        self.tab4_additional_label4.pack(side=tk.LEFT)
        self.tab4_additional_label4.configure(state='disabled')

        self.tab4_additional_field4 = tk.Entry(self.tab4)
        self.tab4_additional_field4.pack(side=tk.LEFT)
        self.tab4_additional_field4.bind("<Return>", lambda event: self.add_additional_values_dl())
        self.tab4_additional_field4.configure(state='disabled')

    def create_tab5_widgets(self):
        self.script5_button = tk.Button(self.tab5)
        self.script5_button["text"] = "Train and Test - wartości domyślne"
        self.script5_button["command"] = self.add_text_dd
        self.script5_button.pack(side=tk.TOP)

        self.tab5_var = tk.IntVar()
        self.tab5_checkbox = tk.Checkbutton(self.tab5, text="Funkcje zaawansowane", variable=self.tab5_var,
                                            command=self.show_additional_functions)
        self.tab5_checkbox.pack(side=tk.TOP)

        self.tab5_additional_label1 = tk.Label(self.tab5, text="Rozmiar tablicy testowej")
        self.tab5_additional_label1.pack(side=tk.LEFT)
        self.tab5_additional_label1.configure(state='disabled')

        self.tab5_additional_field1 = tk.Entry(self.tab5)
        self.tab5_additional_field1.pack(side=tk.LEFT)
        self.tab5_additional_field1.bind("<Return>", lambda event: self.add_additional_values_dd())
        self.tab5_additional_field1.configure(state='disabled')

        self.tab5_additional_label2 = tk.Label(self.tab5, text="Stan losowy")
        self.tab5_additional_label2.pack(side=tk.LEFT)
        self.tab5_additional_label2.configure(state='disabled')

        self.tab5_additional_field2 = tk.Entry(self.tab5)
        self.tab5_additional_field2.pack(side=tk.LEFT)
        self.tab5_additional_field2.bind("<Return>", lambda event: self.add_additional_values_dd())
        self.tab5_additional_field2.configure(state='disabled')

        self.tab5_additional_label3 = tk.Label(self.tab5, text="Maksymalna głębokość")
        self.tab5_additional_label3.pack(side=tk.LEFT)
        self.tab5_additional_label3.configure(state='disabled')

        self.tab5_additional_field3 = tk.Entry(self.tab5)
        self.tab5_additional_field3.pack(side=tk.LEFT)
        self.tab5_additional_field3.bind("<Return>", lambda event: self.add_additional_values_dd())
        self.tab5_additional_field3.configure(state='disabled')

    def create_tab3_widgets(self):
        self.script3_button = tk.Button(self.tab3)
        self.script3_button["text"] = "Grupowanie - wartości domyślne"
        self.script3_button["command"] = self.add_text_gkm
        self.script3_button.pack(side=tk.TOP)

        self.tab3_var = tk.IntVar()
        self.tab3_checkbox = tk.Checkbutton(self.tab3, text="Funkcje zaawansowane", variable=self.tab3_var,
                                            command=self.show_additional_functions)
        self.tab3_checkbox.pack(side=tk.TOP)

        self.tab3_additional_label1 = tk.Label(self.tab3, text="Liczba klastrów")
        self.tab3_additional_label1.pack(side=tk.LEFT)
        self.tab3_additional_label1.configure(state='disabled')

        self.tab3_additional_field1 = tk.Entry(self.tab3)
        self.tab3_additional_field1.pack(side=tk.LEFT)
        self.tab3_additional_field1.bind("<Return>", lambda event: self.add_additional_values_gkm())
        self.tab3_additional_field1.configure(state='disabled')

        self.tab3_additional_label2 = tk.Label(self.tab3, text="Minimalna liczba iteracji")
        self.tab3_additional_label2.pack(side=tk.LEFT)
        self.tab3_additional_label2.configure(state='disabled')

        self.tab3_additional_field2 = tk.Entry(self.tab3)
        self.tab3_additional_field2.pack(side=tk.LEFT)
        self.tab3_additional_field2.bind("<Return>", lambda event: self.add_additional_values_gkm())
        self.tab3_additional_field2.configure(state='disabled')

        self.tab3_additional_label3 = tk.Label(self.tab3, text="Maksymalna liczba iteracji")
        self.tab3_additional_label3.pack(side=tk.LEFT)
        self.tab3_additional_label3.configure(state='disabled')

        self.tab3_additional_field3 = tk.Entry(self.tab3)
        self.tab3_additional_field3.pack(side=tk.LEFT)
        self.tab3_additional_field3.bind("<Return>", lambda event: self.add_additional_values_gkm())
        self.tab3_additional_field3.configure(state='disabled')

        self.tab3_additional_label4 = tk.Label(self.tab3, text="Stan losowy")
        self.tab3_additional_label4.pack(side=tk.LEFT)
        self.tab3_additional_label4.configure(state='disabled')

        self.tab3_additional_field4 = tk.Entry(self.tab3)
        self.tab3_additional_field4.pack(side=tk.LEFT)
        self.tab3_additional_field4.bind("<Return>", lambda event: self.add_additional_values_gkm())
        self.tab3_additional_field4.configure(state='disabled')

    def create_tab6_widgets(self):
        self.script6_button = tk.Button(self.tab6)
        self.script6_button["text"] = "Grupowanie - wartości domyślne"
        self.script6_button["command"] = self.add_text_gdb
        self.script6_button.pack(side=tk.TOP)

        self.tab6_var = tk.IntVar()
        self.tab6_checkbox = tk.Checkbutton(self.tab6, text="Funkcje zaawansowane", variable=self.tab6_var,
                                            command=self.show_additional_functions)
        self.tab6_checkbox.pack(side=tk.TOP)

        self.tab6_additional_label1 = tk.Label(self.tab6, text="Wsparcie")
        self.tab6_additional_label1.pack(side=tk.LEFT)
        self.tab6_additional_label1.configure(state='disabled')

        self.tab6_additional_field1 = tk.Entry(self.tab6)
        self.tab6_additional_field1.pack(side=tk.LEFT)
        self.tab6_additional_field1.bind("<Return>", lambda event: self.add_additional_values_gdb())
        self.tab6_additional_field1.configure(state='disabled')

        self.tab6_additional_label2 = tk.Label(self.tab6, text="Zaufanie")
        self.tab6_additional_label2.pack(side=tk.LEFT)
        self.tab6_additional_label2.configure(state='disabled')

        self.tab6_additional_field2 = tk.Entry(self.tab6)
        self.tab6_additional_field2.pack(side=tk.LEFT)
        self.tab6_additional_field2.bind("<Return>", lambda event: self.add_additional_values_gdb())
        self.tab6_additional_label2.configure(state='disabled')

        self.tab6_menu_label = tk.Label(self.tab6, text="Wybierz metrykę:")
        self.tab6_menu_label.pack(side=tk.LEFT)
        self.tab6_menu_label.configure(state='disabled')

        self.tab6_menu = ttk.Combobox(self.tab6, values=["euclidean", "manhattan", "minkowski"])
        self.tab6_menu.bind("<Return>", lambda event: self.add_additional_values_gdb())
        self.tab6_menu.configure(state='disabled')
        self.tab6_menu.pack(side=tk.LEFT)

    def create_tab7_widgets(self):
        self.script7_button = tk.Button(self.tab7)
        self.script7_button["text"] = "Grupowanie - wartości domyślne"
        self.script7_button["command"] = self.add_text_gh
        self.script7_button.pack(side=tk.TOP)

        self.tab7_var = tk.IntVar()
        self.tab7_checkbox = tk.Checkbutton(self.tab7, text="Funkcje zaawansowane", variable=self.tab7_var,
                                            command=self.show_additional_functions)
        self.tab7_checkbox.pack(side=tk.TOP)

        self.tab7_additional_label1 = tk.Label(self.tab7, text="Liczba klastrów")
        self.tab7_additional_label1.pack(side=tk.LEFT)
        self.tab7_additional_label1.configure(state='disabled')

        self.tab7_additional_field1 = tk.Entry(self.tab7)
        self.tab7_additional_field1.pack(side=tk.LEFT)
        self.tab7_additional_field1.bind("<Return>", lambda event: self.add_additional_values_gh())
        self.tab7_additional_field1.configure(state='disabled')

        self.tab7_menu_label = tk.Label(self.tab7, text="Wybierz metrykę:")
        self.tab7_menu_label.pack(side=tk.LEFT)
        self.tab7_menu_label.configure(state='disabled')

        self.tab7_menu = ttk.Combobox(self.tab7, values=["euclidean", "manhattan", "minkowski"])
        self.tab7_menu.bind("<Return>", lambda event: self.add_additional_values_gh())
        self.tab7_menu.configure(state='disabled')
        self.tab7_menu.pack(side=tk.LEFT)

    def add_text_ad(self):
            self.script_text.insert(tk.END, """
import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

st.header("Analiza reguł asocjacyjnych")

uploaded_file = st.file_uploader("Wybierz plik")  # Uploader plików

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)  # Odczytanie zbioru danych
    st.write(df)  # Wypisanie data frame pliku

    min_support = st.slider('Minimalne wsparcie', min_value=0.0, max_value=1.0, value=0.1, step=0.01)  # Minimalne wsparcie
    min_confidence = st.slider('Minimalne zaufanie', min_value=0.0, max_value=1.0, value=0.5, step=0.01)  # Minimalne zaufanie

    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

    # Filtruj reguły według minimalnego zaufania
    rules = rules[rules['confidence'] >= min_confidence]

    st.write("Liczba reguł znalezionych: ", len(rules))

    st.write("Najważniejsze reguły:")
    st.table(rules)
""")

    def add_text_knn(self):
        self.script_text.insert(tk.END, """
import streamlit as st
import numpy as np
import pandas as pd
from sklearn import metrics, datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

st.header("Analiza klasyfikatora KNN")

uploaded_file = st.file_uploader("Wybierz plik")   #Uploader plików
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file) #Odczytanie zbioru danych
    st.write(df)   #Wypisanie data frame pliku
    column_list = list(df.columns)
    selected_columns = st.multiselect("Wybierz kolumny", column_list[:-1],
                                      default=[column_list[0], column_list[1]])  # Wybór kolumn dla cech
    selected_labels = st.multiselect("Wybierz kolumny", column_list[-1:], default=[column_list[-1]])  # Wybór kolumn dla etykiet

    features = df[selected_columns]  # Wyodrębnienie części warunkowej danych
    labels = df[selected_labels]  # Wyodrębnienie kolumny decyzyjnej

    ts = st.number_input('Rozmiar tablicy testowej', min_value=0.0, max_value=1.0, value=0.6, step=0.01) #Ustalenie tablicy treningowej
    rs = st.number_input('Stan losowy', min_value=1, max_value=10000, value=1234, step=1) #Ustalenie ziarna generatora liczb pseudolosowych

    datasets = train_test_split(features, labels, test_size=ts, random_state=rs)

    features_train = datasets[0]
    features_test = datasets[1]
    labels_train = datasets[2]
    labels_test = datasets[3]

    nm = st.number_input('Liczba sąsiadów', min_value=1, max_value=10000, value=5, step=1)  #Liczba sąsiadów
    myNoNeighbors = nm

    choice = st.selectbox(  #Checkbox do wyboru metryki

        'Wybierz jedną z dostępnych metryk',

        ('euclidean', 'manhattan', 'minkowski'))
    myMetric = choice

    model = KNeighborsClassifier(n_neighbors=myNoNeighbors, metric=myMetric)  #Utworzenie obiektu przykładowego modelu klasyfikatora (k-NN)
    model.fit(features_train, np.ravel(labels_train)) #Uczenie klasyfikatora na części treningowej

    labels_predicted = model.predict(features_test) #Generowania decyzji dla części testowej

    accuracy = metrics.accuracy_score(labels_test, labels_predicted)  #Policzenie jakości klasyfikacji

    st.write("Classification accuracy=", accuracy)
    st.write("========= PEŁNE WYNIKI KLASYFIKACJI ================")
    report = classification_report(labels_test, labels_predicted, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.write(df_report)

    st.write("====== Tablica pomyłek =========")
    conf_matrix = confusion_matrix(labels_test, labels_predicted)
    df_conf_matrix = pd.DataFrame(conf_matrix)
    st.write(df_conf_matrix)
    """)

    def add_text_dl(self):
        self.script_text.insert(tk.END, """
import streamlit as st
import numpy as np
import pandas as pd
from sklearn import metrics, datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

st.header("Analiza klasyfikatora drzewa losowego")


uploaded_file = st.file_uploader("Wybierz plik")   #Uploader plików
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file) #Odczytanie zbioru danych
    st.write(df)   #Wypisanie data frame pliku
    column_list = list(df.columns)
    selected_columns = st.multiselect("Wybierz kolumny", column_list[:-1],
                                      default=[column_list[0], column_list[1]])  # Wybór kolumn dla cech
    selected_labels = st.multiselect("Wybierz kolumny", column_list[-1:], default=[column_list[-1]])  # Wybór kolumn dla etykiet

    features = df[selected_columns]  # Wyodrębnienie części warunkowej danych
    labels = df[selected_labels]  # Wyodrębnienie kolumny decyzyjnej

    ts = st.number_input('Rozmiar tablicy testowej', min_value=0.0, max_value=1.0, value=0.6, step=0.01) #Ustalenie tablicy treningowej
    rs = st.number_input('Stan losowy', min_value=1, max_value=10000, value=1234, step=1) #Ustalenie ziarna generatora liczb pseudolosowych

    datasets = train_test_split(features, labels, test_size=ts, random_state=rs)

    features_train = datasets[0]
    features_test = datasets[1]
    labels_train = datasets[2]
    labels_test = datasets[3]

    n_estimators = st.number_input('Liczba estymatorów', min_value=1, max_value=10000, value=100, step=1)  #Liczba estymatorów w drzewie losowym
    max_depth = st.number_input('Maksymalna głębokość', min_value=1, max_value=10000, value=5, step=1)  #Maksymalna głębokość drzewa

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)  #Utworzenie obiektu klasyfikatora drzewa losowego
    model.fit(features_train, np.ravel(labels_train)) #Uczenie klasyfikatora na części treningowej

    labels_predicted = model.predict(features_test) #Generowanie decyzji dla części testowej

    accuracy = metrics.accuracy_score(labels_test, labels_predicted)  #Policzenie jakości klasyfikacji

    st.write("Classification accuracy=", accuracy)
    st.write("========= PEŁNE WYNIKI KLASYFIKACJI ================")
    report = classification_report(labels_test, labels_predicted, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.write(df_report)

    st.write("====== Tablica pomyłek =========")
    conf_matrix = confusion_matrix(labels_test, labels_predicted)
    df_conf_matrix = pd.DataFrame(conf_matrix)
    st.write(df_conf_matrix)
        """)

    def add_text_dd(self):
        self.script_text.insert(tk.END, """
import streamlit as st
import numpy as np
import pandas as pd
from sklearn import metrics, datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

st.header("Analiza klasyfikatora drzewa decyzyjnego")

uploaded_file = st.file_uploader("Wybierz plik")   #Uploader plików
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file) #Odczytanie zbioru danych
    st.write(df)   #Wypisanie data frame pliku
    column_list = list(df.columns)
    selected_columns = st.multiselect("Wybierz kolumny", column_list[:-1],
                                      default=[column_list[0], column_list[1]])  # Wybór kolumn dla cech
    selected_labels = st.multiselect("Wybierz kolumny", column_list[-1:], default=[column_list[-1]])  # Wybór kolumn dla etykiet

    features = df[selected_columns]  # Wyodrębnienie części warunkowej danych
    labels = df[selected_labels]  # Wyodrębnienie kolumny decyzyjnej

    ts = st.number_input('Rozmiar tablicy testowej', min_value=0.0, max_value=1.0, value=0.6, step=0.01) #Ustalenie tablicy treningowej
    rs = st.number_input('Stan losowy', min_value=1, max_value=10000, value=1234, step=1) #Ustalenie ziarna generatora liczb pseudolosowych

    datasets = train_test_split(features, labels, test_size=ts, random_state=rs)

    features_train = datasets[0]
    features_test = datasets[1]
    labels_train = datasets[2]
    labels_test = datasets[3]

    max_depth = st.number_input('Maksymalna głębokość', min_value=1, max_value=10000, value=5, step=1)  #Maksymalna głębokość drzewa

    model = DecisionTreeClassifier(max_depth=max_depth)  #Utworzenie obiektu klasyfikatora drzewa decyzyjnego
    model.fit(features_train, np.ravel(labels_train)) #Uczenie klasyfikatora na części treningowej

    labels_predicted = model.predict(features_test) #Generowanie decyzji dla części testowej

    accuracy = metrics.accuracy_score(labels_test, labels_predicted)  #Policzenie jakości klasyfikacji

    st.write("Classification accuracy=", accuracy)
    st.write("========= PEŁNE WYNIKI KLASYFIKACJI ================")
    report = classification_report(labels_test, labels_predicted, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.write(df_report)

    st.write("====== Tablica pomyłek =========")
    conf_matrix = confusion_matrix(labels_test, labels_predicted)
    df_conf_matrix = pd.DataFrame(conf_matrix)
    st.write(df_conf_matrix)
        """)

    def add_text_gkm(self):
        self.script_text.insert(tk.END, """
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.header("Analiza grupowania KMeans")

uploaded_file = st.file_uploader("Wybierz plik")  # Uploader plików

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)  # Odczytanie zbioru danych
    st.write(df)  # Wypisanie data frame pliku

    column_list = list(df.columns)
    selected_columns = st.multiselect("Wybierz kolumny do grupowania", column_list, default=[column_list[0], column_list[3]])

    if len(selected_columns) >= 2:
        features = df[selected_columns]  # Wybór kolumn do grupowania
        sFeatures = features
        scaler = StandardScaler()
        sFeatures = scaler.fit_transform(features)

        nc = st.number_input('Liczba klastrów', min_value=0, max_value=1000, value=4,
                                 step=1)  # Wybór liczby klastór
        ni = st.number_input('Minimalna liczba iteracji', min_value=0, max_value=1000, value=10,
                                 step=1)  # Wybór minimalnej liczby iteracji
        mi = st.number_input('Maksymalna liczba iteracji', min_value=0, max_value=1000, value=1000,
                                 step=1)  # Wybór maksymalnej liczby iteracji
        rs = st.number_input('Stan losowy', min_value=0, max_value=10000, value=1234,
                                 step=1)  # Ustalenie ziarna generatora liczb pseudolosowych

        kmeans = KMeans(n_clusters=nc, init='k-means++', n_init=ni, max_iter=mi, random_state=rs)
        kmeans.fit(sFeatures)  # Grupowanie

        # Wizualizacja grupowania
        centroidsKMeans = kmeans.cluster_centers_
        centroidsKMeansX = centroidsKMeans[:, 0]
        centroidsKMeansY = centroidsKMeans[:, 1]
        clusters = kmeans.fit_predict(features)

        x = sFeatures[:, 0]
        y = sFeatures[:, 1]

        fig, ax = plt.subplots()
        scatter = ax.scatter(x, y, s=10, c=clusters, alpha=0.9)
        centroids = ax.scatter(centroidsKMeansX, centroidsKMeansY, s=50, color="blue", alpha=0.9)

        # Dodanie etykiet wybranych kolumn
        ax.set_xlabel(selected_columns[0])
        ax.set_ylabel(selected_columns[1])

        ax.legend([scatter, centroids], ['Data Points', 'Centroids'])
        st.pyplot(fig)

        st.write("Przyporządkowanie poszczególnych obiektów do skupień:")
        data = {"Skupienie": [], "Obiekty": []}
        for i in range(nc):
            cluster_indices = [index for index, cluster in enumerate(clusters) if cluster == i]
            objects = ", ".join([str(index + 1) for index in cluster_indices])
            data["Skupienie"].append(i)
            data["Obiekty"].append(objects)
        cluster_data = pd.DataFrame(data)
        st.table(cluster_data)
    else:
        st.write("Wybierz co najmniej dwie kolumny do grupowania.")
        """)

    def add_text_gdb(self):
        self.script_text.insert(tk.END, """
from sklearn.cluster import DBSCAN
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.header("Analiza grupowania DBSCAN")

uploaded_file = st.file_uploader("Wybierz plik", key='2')  # Uploader plików

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)  # Odczytanie zbioru danych
    st.write(df)  # Wypisanie data frame pliku

    column_list = list(df.columns)
    selected_columns = st.multiselect("Wybierz kolumny do grupowania", column_list,default=[column_list[0], column_list[3]])

    if len(selected_columns) >= 2:
        features = df[selected_columns]  # Wybór kolumn do grupowania
        e = st.number_input('Maksymalna odległość między obserwacjami, które należy uznać za sąsiadujące.', min_value=0,
                                max_value=1000, value=3, step=1)
        ms = st.number_input('Minimalna liczba próbek', min_value=0, max_value=1000, value=5, step=1)
        met = st.selectbox('Wybierz jedną z dostępnych metryk', ('euclidean', 'manhattan', 'minkowski'))

        db = DBSCAN(eps=e, min_samples=ms, metric=met)
        db.fit(features)

        clusters = db.fit_predict(features)

        x = np.ravel(features.iloc[:, [0]])
        y = np.ravel(features.iloc[:, [1]])

        fig, ax = plt.subplots()
        scatter = ax.scatter(x, y, s=10, c=clusters, alpha=0.9)

        # Dodanie etykiet wybranych kolumn
        ax.set_xlabel(selected_columns[0])
        ax.set_ylabel(selected_columns[1])

        ax.legend([scatter], ['Data Points'])
        st.pyplot(fig)

        st.write("Przyporządkowanie poszczególnych obiektów do skupień:")
        data = {"Skupienie": [], "Obiekty": []}
        for i in range(len(np.unique(clusters))):
            cluster_indices = [index for index, cluster in enumerate(clusters) if cluster == i]
            objects = ", ".join([str(index + 1) for index in cluster_indices])
            data["Skupienie"].append(i)
            data["Obiekty"].append(objects)
        cluster_data = pd.DataFrame(data)
        st.table(cluster_data)
    else:
        st.write("Wybierz co najmniej dwie kolumny do grupowania.")
        """)

    def add_text_gh(self):
        self.script_text.insert(tk.END, """
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

st.header("Analiza grupowania hierarchicznego")

uploaded_file = st.file_uploader("Wybierz plik", key='3')  # Uploader plików

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)  # Odczytanie zbioru danych
    st.write(df)  # Wypisanie data frame pliku

    column_list = list(df.columns)
    selected_columns = st.multiselect("Wybierz kolumny do grupowania", column_list, default=[column_list[0], column_list[3]], key='4')

    if len(selected_columns) >= 2:
        features = df[selected_columns]  # Wybór kolumn do grupowania

        nc = st.number_input('Liczba klastrów', min_value=0, max_value=1000, value=4,
                             step=1, key='num_clusters')  # Wybór liczby klastrów
        met = st.selectbox('Wybierz jedną z dostępnych metryk', ('euclidean', 'manhattan', 'minkowski'))

        ac = AgglomerativeClustering(n_clusters=nc, affinity=met, linkage=('average'))
        ac.fit(features)

        clusters = ac.fit_predict(features)

        x = np.ravel(features.iloc[:, [0]])
        y = np.ravel(features.iloc[:, [1]])

        fig, ax = plt.subplots()
        scatter = ax.scatter(x, y, s=10, c=clusters, alpha=0.9)

        # Dodanie etykiet wybranych kolumn
        ax.set_xlabel(selected_columns[0])
        ax.set_ylabel(selected_columns[1])

        ax.legend([scatter], ['Data Points'])
        st.pyplot(fig)

        st.write("Przyporządkowanie poszczególnych obiektów do skupień:")
        data = {"Skupienie": [], "Obiekty": []}
        for i in range(nc):
            cluster_indices = [index for index, cluster in enumerate(clusters) if cluster == i]
            objects = ", ".join([str(index + 1) for index in cluster_indices])
            data["Skupienie"].append(i)
            data["Obiekty"].append(objects)
        cluster_data = pd.DataFrame(data)
        st.table(cluster_data)
    else:
        st.write("Wybierz co najmniej dwie kolumny do grupowania.")
        """)

    def show_additional_functions(self):

        if self.tab1_var.get():
            self.tab1_additional_label1.configure(state='normal')
            self.tab1_additional_label2.configure(state='normal')
            self.tab1_additional_field1.configure(state='normal')
            self.tab1_additional_field2.configure(state='normal')
        else:
            self.tab1_additional_label1.configure(state='disabled')
            self.tab1_additional_label2.configure(state='disabled')
            self.tab1_additional_field1.configure(state='disabled')
            self.tab1_additional_field2.configure(state='disabled')

        if self.tab2_var.get():
            self.tab2_additional_label1.configure(state='normal')
            self.tab2_additional_label2.configure(state='normal')
            self.tab2_additional_label3.configure(state='normal')
            self.tab2_additional_field1.configure(state='normal')
            self.tab2_additional_field2.configure(state='normal')
            self.tab2_additional_field3.configure(state='normal')
            self.tab2_menu_label.configure(state='normal')
            self.tab2_menu.configure(state='normal')
        else:
            self.tab2_additional_label1.configure(state='disabled')
            self.tab2_additional_label2.configure(state='disabled')
            self.tab2_additional_label3.configure(state='disabled')
            self.tab2_additional_field1.configure(state='disabled')
            self.tab2_additional_field2.configure(state='disabled')
            self.tab2_additional_field3.configure(state='disabled')
            self.tab2_menu_label.configure(state='disabled')
            self.tab2_menu.configure(state='disabled')

        if self.tab4_var.get():
            self.tab4_additional_label1.configure(state='normal')
            self.tab4_additional_label2.configure(state='normal')
            self.tab4_additional_label3.configure(state='normal')
            self.tab4_additional_label4.configure(state='normal')
            self.tab4_additional_field1.configure(state='normal')
            self.tab4_additional_field2.configure(state='normal')
            self.tab4_additional_field3.configure(state='normal')
            self.tab4_additional_field4.configure(state='normal')
        else:
            self.tab4_additional_label1.configure(state='disabled')
            self.tab4_additional_label2.configure(state='disabled')
            self.tab4_additional_label3.configure(state='disabled')
            self.tab4_additional_label4.configure(state='disabled')
            self.tab4_additional_field1.configure(state='disabled')
            self.tab4_additional_field2.configure(state='disabled')
            self.tab4_additional_field3.configure(state='disabled')
            self.tab4_additional_field4.configure(state='disabled')

        if self.tab5_var.get():
            self.tab5_additional_label1.configure(state='normal')
            self.tab5_additional_label2.configure(state='normal')
            self.tab5_additional_label3.configure(state='normal')
            self.tab5_additional_field1.configure(state='normal')
            self.tab5_additional_field2.configure(state='normal')
            self.tab5_additional_field3.configure(state='normal')
        else:
            self.tab5_additional_label1.configure(state='disabled')
            self.tab5_additional_label2.configure(state='disabled')
            self.tab5_additional_label3.configure(state='disabled')
            self.tab5_additional_field1.configure(state='disabled')
            self.tab5_additional_field2.configure(state='disabled')
            self.tab5_additional_field3.configure(state='disabled')

        if self.tab3_var.get():
            self.tab3_additional_label1.configure(state='normal')
            self.tab3_additional_label2.configure(state='normal')
            self.tab3_additional_label3.configure(state='normal')
            self.tab3_additional_label4.configure(state='normal')
            self.tab3_additional_field1.configure(state='normal')
            self.tab3_additional_field2.configure(state='normal')
            self.tab3_additional_field3.configure(state='normal')
            self.tab3_additional_field4.configure(state='normal')
        else:
            self.tab3_additional_label1.configure(state='disabled')
            self.tab3_additional_label2.configure(state='disabled')
            self.tab3_additional_label3.configure(state='disabled')
            self.tab3_additional_label4.configure(state='disabled')
            self.tab3_additional_field1.configure(state='disabled')
            self.tab3_additional_field2.configure(state='disabled')
            self.tab3_additional_field3.configure(state='disabled')
            self.tab3_additional_field4.configure(state='disabled')

        if self.tab6_var.get():
            self.tab6_additional_label1.configure(state='normal')
            self.tab6_additional_label2.configure(state='normal')
            self.tab6_additional_field1.configure(state='normal')
            self.tab6_additional_field2.configure(state='normal')
            self.tab6_menu_label.configure(state='normal')
            self.tab6_menu.configure(state='normal')
        else:
            self.tab6_additional_label1.configure(state='disabled')
            self.tab6_additional_field1.configure(state='disabled')
            self.tab6_additional_label2.configure(state='disabled')
            self.tab6_additional_field2.configure(state='disabled')
            self.tab6_menu_label.configure(state='disabled')
            self.tab6_menu.configure(state='disabled')

        if self.tab7_var.get():
            self.tab7_additional_label1.configure(state='normal')
            self.tab7_additional_field1.configure(state='normal')
            self.tab7_menu_label.configure(state='normal')
            self.tab7_menu.configure(state='normal')

        else:
            self.tab7_additional_label1.configure(state='disabled')
            self.tab7_additional_field1.configure(state='disabled')
            self.tab7_menu_label.configure(state='disabled')
            self.tab7_menu.configure(state='disabled')

        self.script_text.delete("1.0", tk.END)

    def add_additional_values_a(self, event=None):
        support = self.tab1_additional_field1.get()
        if not support:  # Jeśli wartość support nie została podana
            support = "0.15"  # Przypisz wartość domyślną

        confidence = self.tab1_additional_field2.get()
        if not confidence:  # Jeśli wartość confidence nie została podana
            confidence = "0.5"  # Przypisz wartość domyślną

        additional_text = """
import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

st.header("Analiza reguł asocjacyjnych")
uploaded_file = st.file_uploader("Wybierz plik")  # Uploader plików

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)  # Odczytanie zbioru danych
    st.write(df)  # Wypisanie data frame pliku
    """
        self.script_text.insert(tk.END, additional_text)

        additional_values = f"min_support = {support}\n    min_confidence = {confidence}\n"

        self.script_text.insert(tk.END, additional_values)
        additional_text = """
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

    # Filtruj reguły według minimalnego zaufania
    rules = rules[rules['confidence'] >= min_confidence]

    st.write("Liczba reguł znalezionych: ", len(rules))

    st.write("Najważniejsze reguły:")
    st.table(rules)        """
        self.script_text.insert(tk.END, additional_text)
        self.tab1_additional_field1.delete(0, tk.END)
        self.tab1_additional_field2.delete(0, tk.END)

    def add_additional_values_knn(self, event=None):
        ts = self.tab2_additional_field1.get()
        if not ts:
            ts = "0.6"
        rs = self.tab2_additional_field2.get()
        if not rs:
            rs = "1234"
        nm = self.tab2_additional_field3.get()
        if not nm:
            nm = "5"
        choice = self.tab2_menu.get()
        if not choice:
            choice = "euclidean"

        additional_text = """
import streamlit as st
import numpy as np
import pandas as pd
from sklearn import metrics, datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

st.header("Analiza klasyfikatora KNN")

uploaded_file = st.file_uploader("Wybierz plik")   #Uploader plików
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file) #Odczytanie zbioru danych
    st.write(df)   #Wypisanie data frame pliku
    column_list = list(df.columns)
    selected_columns = st.multiselect("Wybór kolumn dla cech", column_list[:-1], default=[column_list[0], column_list[1]])  # Wybór kolumn dla cech
    selected_labels = st.multiselect("Wybór kolumn dla etykiet", column_list[-1:], default=[column_list[-1]])  # Wybór kolumn dla etykiet
    features = df[selected_columns]  # Wyodrębnienie części warunkowej danych
    labels = df[selected_labels]  # Wyodrębnienie kolumny decyzyjnej
"""
        self.script_text.insert(tk.END, additional_text)

        additional_values = f"    ts = {ts}\n    rs = {rs}\n"
        self.script_text.insert(tk.END, additional_values)
        additional_text = """
    datasets = train_test_split(features, labels, test_size=ts, random_state=rs)

    features_train = datasets[0]
    features_test = datasets[1]
    labels_train = datasets[2]
    labels_test = datasets[3]
"""
        self.script_text.insert(tk.END, additional_text)

        additional_values = f"    nm = {nm}\n"
        self.script_text.insert(tk.END, additional_values)

        additional_values = f"    myMetric = '{choice}'\n"
        self.script_text.insert(tk.END, additional_values)

        additional_text = """
    #Utworzenie obiektu przykładowego modelu klasyfikatora (k-NN)
    model = KNeighborsClassifier(n_neighbors=nm, metric=myMetric)  
    #Uczenie klasyfikatora na części treningowej
    model.fit(features_train, np.ravel(labels_train)) 

    labels_predicted = model.predict(features_test) #Generowania decyzji dla części testowej

    accuracy = metrics.accuracy_score(labels_test, labels_predicted)  #Policzenie jakości klasyfikacji

    st.write("Classification accuracy=", accuracy)
    st.write("========= PEŁNE WYNIKI KLASYFIKACJI ================")
    report = classification_report(labels_test, labels_predicted, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.write(df_report)

    st.write("====== Tablica pomyłek =========")
    conf_matrix = confusion_matrix(labels_test, labels_predicted)
    df_conf_matrix = pd.DataFrame(conf_matrix)
    st.write(df_conf_matrix)
                    """
        self.script_text.insert(tk.END, additional_text)

        self.tab2_additional_field1.delete(0, tk.END)
        self.tab2_additional_field2.delete(0, tk.END)

    def add_additional_values_dl(self, event=None):
        ts = self.tab4_additional_field1.get()
        if not ts:
            ts= "0.5"
        rs = self.tab4_additional_field2.get()
        if not rs:
            rs= "1234"
        ne = self.tab4_additional_field3.get()
        if not ne:
            ne= "100"
        md = self.tab4_additional_field4.get()
        if not md:
            md= "5"
        additional_text = """
import streamlit as st
import numpy as np
import pandas as pd
from sklearn import metrics, datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

st.header("Analiza klasyfikatora drzewa losowego")

uploaded_file = st.file_uploader("Wybierz plik")  # Uploader plików
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)  # Odczytanie zbioru danych
    st.write(df)  # Wypisanie data frame pliku
    column_list = list(df.columns)
    selected_columns = st.multiselect("Wybór kolumn dla cech", column_list[:-1], default=[column_list[0], column_list[1]])  # Wybór kolumn dla cech
    selected_labels = st.multiselect("Wybór kolumn dla etykiet", column_list[-1:], default=[column_list[-1]])  # Wybór kolumn dla etykiet

    features = df[selected_columns]  # Wyodrębnienie części warunkowej danych
    labels = df[selected_labels]  # Wyodrębnienie kolumny decyzyjnej

"""
        self.script_text.insert(tk.END, additional_text)

        additional_values = f"    ts = {ts}\n    rs = {rs}\n"
        self.script_text.insert(tk.END, additional_values)
        additional_text = """
    datasets = train_test_split(features, labels, test_size=ts, random_state=rs)

    features_train = datasets[0]
    features_test = datasets[1]
    labels_train = datasets[2]
    labels_test = datasets[3]
"""
        self.script_text.insert(tk.END, additional_text)

        additional_values = f"    n_estimators = {ne}\n    max_depth = {md}\n"
        self.script_text.insert(tk.END, additional_values)

        additional_text = """
    #Utworzenie obiektu klasyfikatora drzewa losowego
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    #Uczenie klasyfikatora na części treningowej  
    model.fit(features_train, np.ravel(labels_train)) 

    labels_predicted = model.predict(features_test) #Generowanie decyzji dla części testowej

    accuracy = metrics.accuracy_score(labels_test, labels_predicted)  #Policzenie jakości klasyfikacji

    st.write("Classification accuracy=", accuracy)
    st.write("========= PEŁNE WYNIKI KLASYFIKACJI ================")
    report = classification_report(labels_test, labels_predicted, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.write(df_report)

    st.write("====== Tablica pomyłek =========")
    conf_matrix = confusion_matrix(labels_test, labels_predicted)
    df_conf_matrix = pd.DataFrame(conf_matrix)
    st.write(df_conf_matrix)
        """
        self.script_text.insert(tk.END, additional_text)

    def add_additional_values_dd(self, event=None):
            ts = self.tab5_additional_field1.get()
            if not ts:
                ts = "0.6"
            rs = self.tab5_additional_field2.get()
            if not rs:
                rs = "1234"
            md = self.tab5_additional_field3.get()
            if not md:
                md = "5"
            additional_text = """
import streamlit as st
import numpy as np
import pandas as pd
from sklearn import metrics, datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

st.header("Analiza klasyfikatora drzewa decyzyjnego")

uploaded_file = st.file_uploader("Wybierz plik")   #Uploader plików
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file) #Odczytanie zbioru danych
    st.write(df)   #Wypisanie data frame pliku
    column_list = list(df.columns)
    # Wybór kolumn dla cech
    selected_columns = st.multiselect("Wybór kolumn dla cech", column_list[:-1],
                                      default=[column_list[0], column_list[1]])  
    # Wybór kolumn dla etykiet
    selected_labels = st.multiselect("Wybór kolumn dla etykiet", column_list[-1:], default=[column_list[-1]])  

    features = df[selected_columns]  # Wyodrębnienie części warunkowej danych
    labels = df[selected_labels]  # Wyodrębnienie kolumny decyzyjnej
"""
            self.script_text.insert(tk.END, additional_text)

            additional_values = f"    ts = {ts}\n    rs = {rs}\n"
            self.script_text.insert(tk.END, additional_values)
            additional_text = """
    datasets = train_test_split(features, labels, test_size=ts, random_state=rs)

    features_train = datasets[0]
    features_test = datasets[1]
    labels_train = datasets[2]
    labels_test = datasets[3]
"""
            self.script_text.insert(tk.END, additional_text)

            additional_values = f"    max_depth = {md}\n    "
            self.script_text.insert(tk.END, additional_values)

            additional_text = """
    model = DecisionTreeClassifier(max_depth=max_depth)  #Utworzenie obiektu klasyfikatora drzewa decyzyjnego
    model.fit(features_train, np.ravel(labels_train)) #Uczenie klasyfikatora na części treningowej

    labels_predicted = model.predict(features_test) #Generowanie decyzji dla części testowej

    accuracy = metrics.accuracy_score(labels_test, labels_predicted)  #Policzenie jakości klasyfikacji

    st.write("Classification accuracy=", accuracy)
    st.write("========= PEŁNE WYNIKI KLASYFIKACJI ================")
    report = classification_report(labels_test, labels_predicted, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.write(df_report)

    st.write("====== Tablica pomyłek =========")
    conf_matrix = confusion_matrix(labels_test, labels_predicted)
    df_conf_matrix = pd.DataFrame(conf_matrix)
    st.write(df_conf_matrix)
            """
            self.script_text.insert(tk.END, additional_text)

    def add_additional_values_gkm(self, event=None):
        nc = self.tab3_additional_field1.get()
        if not nc:
            nc = "4"
        ni = self.tab3_additional_field2.get()
        if not ni:
            ni = "10"
        mi = self.tab3_additional_field3.get()
        if not mi:
            mi = "1000"
        rs = self.tab3_additional_field4.get()
        if not rs:
            rs = "1234"
        additional_text = """
from sklearn.cluster import KMeans
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

st.header("Analiza grupowania KMeans")

uploaded_file = st.file_uploader("Wybierz plik")  # Uploader plików

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)  # Odczytanie zbioru danych
    st.write(df)  # Wypisanie data frame pliku

    column_list = list(df.columns)
    selected_columns = st.multiselect("Wybierz kolumny do grupowania", column_list, 
    default=[column_list[0], column_list[3]])
    if len(selected_columns) >= 2:
        features = df[selected_columns]  # Wybór kolumn do grupowania
        sFeatures = features
        scaler = StandardScaler()
        sFeatures = scaler.fit_transform(features)
"""

        self.script_text.insert(tk.END, additional_text)
        additional_values = f"        nc = {nc}\n        ni = {ni}\n        mi = {mi}\n        rs = {rs}\n"
        self.script_text.insert(tk.END, additional_values)

        additional_text = """
        kmeans = KMeans(n_clusters=nc, init='k-means++', n_init=ni, max_iter=mi, random_state=rs)
        kmeans.fit(sFeatures)  # Grupowanie

        # Wizualizacja grupowania
        centroidsKMeans = kmeans.cluster_centers_
        centroidsKMeansX = centroidsKMeans[:, 0]
        centroidsKMeansY = centroidsKMeans[:, 1]
        clusters = kmeans.fit_predict(features)

        x = sFeatures[:, 0]
        y = sFeatures[:, 1]

        fig, ax = plt.subplots()
        scatter = ax.scatter(x, y, s=10, c=clusters, alpha=0.9)
        centroids = ax.scatter(centroidsKMeansX, centroidsKMeansY, s=50, color="blue", alpha=0.9)

        # Dodanie etykiet wybranych kolumn
        ax.set_xlabel(selected_columns[0])
        ax.set_ylabel(selected_columns[1])

        ax.legend([scatter, centroids], ['Data Points', 'Centroids'])
        st.pyplot(fig)

        st.write("Przyporządkowanie poszczególnych obiektów do skupień:")
        data = {"Skupienie": [], "Obiekty": []}
        for i in range(nc):
            cluster_indices = [index for index, cluster in enumerate(clusters) if cluster == i]
            objects = ", ".join([str(index + 1) for index in cluster_indices])
            data["Skupienie"].append(i)
            data["Obiekty"].append(objects)
        cluster_data = pd.DataFrame(data)
        st.table(cluster_data)
    else:
        st.write("Wybierz co najmniej dwie kolumny do grupowania.")
        """

        self.script_text.insert(tk.END, additional_text)

        self.tab3_additional_field1.delete(0, tk.END)
        self.tab3_additional_field2.delete(0, tk.END)

    def add_additional_values_gdb(self, event=None):
        e = self.tab6_additional_field1.get()
        if not e:
            e = "3"
        ms = self.tab6_additional_field2.get()
        if not ms:
            ms = "5"
        met = self.menu.get()
        if not met:
            met = "3"
        additional_text = """
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

st.header("Analiza grupowania DBSCAN)

uploaded_file = st.file_uploader("Wybierz plik")  # Uploader plików

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)  # Odczytanie zbioru danych
    st.write(df)  # Wypisanie data frame pliku

    column_list = list(df.columns)
    selected_columns = st.multiselect("Wybierz kolumny do grupowania", column_list, 
    default=[column_list[0], column_list[3]])
    if len(selected_columns) >= 2:
        features = df[selected_columns]  # Wybór kolumn do grupowania
"""
        self.script_text.insert(tk.END, additional_text)

        additional_values = f"        e = {e}\n        ms = {ms}\n"
        self.script_text.insert(tk.END, additional_values)

        additional_values = f"        met = '{met}'\n"
        self.script_text.insert(tk.END, additional_values)

        additional_text = """
        db = DBSCAN(eps=e, min_samples=ms, metric=met)
        db.fit(features)

        clusters = db.fit_predict(features)

        x = np.ravel(features.iloc[:, [0]])
        y = np.ravel(features.iloc[:, [1]])

        fig, ax = plt.subplots()
        scatter = ax.scatter(x, y, s=10, c=clusters, alpha=0.9)

        # Dodanie etykiet wybranych kolumn
        ax.set_xlabel(selected_columns[0])
        ax.set_ylabel(selected_columns[1])

        ax.legend([scatter], ['Data Points'])
        st.pyplot(fig)

        st.write("Przyporządkowanie poszczególnych obiektów do skupień:")
        data = {"Skupienie": [], "Obiekty": []}
        for i in range(len(np.unique(clusters))):
            cluster_indices = [index for index, cluster in enumerate(clusters) if cluster == i]
            objects = ", ".join([str(index + 1) for index in cluster_indices])
            data["Skupienie"].append(i)
            data["Obiekty"].append(objects)
        cluster_data = pd.DataFrame(data)
        st.table(cluster_data)
    else:
        st.write("Wybierz co najmniej dwie kolumny do grupowania.")
"""
        self.script_text.insert(tk.END, additional_text)

    def add_additional_values_gh(self, event=None):
        nc = self.tab7_additional_field1.get()
        met = self.menu.get()

        additional_text = """
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

st.header("Analiza grupowania hierarchicznego")

uploaded_file = st.file_uploader("Wybierz plik", key='3')  # Uploader plików

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)  # Odczytanie zbioru danych
    st.write(df)  # Wypisanie data frame pliku

    column_list = list(df.columns)
    selected_columns = st.multiselect("Wybierz kolumny do grupowania", column_list, key='4')
    if len(selected_columns) >= 2:
        features = df[selected_columns]  # Wybór kolumn do grupowania
"""
        self.script_text.insert(tk.END, additional_text)

        additional_values = f"        nc = {nc}\n        met = '{met}'\n"

        self.script_text.insert(tk.END, additional_values)

        additional_text = """
        ac = AgglomerativeClustering(n_clusters=nc, affinity=met,linkage=('average'))
        ac.fit(features)

        clusters = ac.fit_predict(features)

        x = np.ravel(features.iloc[:, [0]])
        y = np.ravel(features.iloc[:, [1]])

        fig, ax = plt.subplots()
        scatter = ax.scatter(x, y, s=10, c=clusters, alpha=0.9)

        # Dodanie etykiet wybranych kolumn
        ax.set_xlabel(selected_columns[0])
        ax.set_ylabel(selected_columns[1])

        ax.legend([scatter], ['Data Points'])
        st.pyplot(fig)

        st.write("Przyporządkowanie poszczególnych obiektów do skupień:")
        data = {"Skupienie": [], "Obiekty": []}
        for i in range(nc):
            cluster_indices = [index for index, cluster in enumerate(clusters) if cluster == i]
            objects = ", ".join([str(index + 1) for index in cluster_indices])
            data["Skupienie"].append(i)
            data["Obiekty"].append(objects)
        cluster_data = pd.DataFrame(data)
        st.table(cluster_data)
    else:
        st.write("Wybierz co najmniej dwie kolumny do grupowania.")        
"""
        self.script_text.insert(tk.END, additional_text)
    def save_script(self):
        script = self.script_text.get("1.0", tk.END)
        filename = filedialog.asksaveasfilename(defaultextension=".py", initialfile="skrypt")
        if filename:
            with open(filename, "w", encoding='utf-8') as f:
                f.write(script)

    def clear_script(self):
        self.script_text.delete("1.0", tk.END)

    def run_script(self):
        script = self.script_text.get("1.0", tk.END)
        filename = "temp_script.py"
        with open(filename, "w", encoding='utf-8') as f:
            f.write(script)
        subprocess.Popen(["streamlit", "run", filename])
        self.clear_script()

root = tk.Tk()
app = Application(master=root)
app.mainloop()