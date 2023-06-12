from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt

uploaded_file = st.file_uploader("Choose a file")  # Uploader plików

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)  # Odczytanie zbioru danych
    st.write(df)  # Wypisanie data frame pliku

    column_list = list(df.columns)
    selected_columns = st.multiselect("Select columns for clustering", column_list, default=[column_list[0], column_list[3]])

    if len(selected_columns) >= 2:
        features = df[selected_columns]  # Wybór kolumn do grupowania
        sFeatures = features
        scaler = StandardScaler()
        sFeatures = scaler.fit_transform(features)

        nc = st.number_input('Numbers of clusters', min_value=0, max_value=1000, value=4,
                                 step=1)  # Wybór liczby klastór
        ni = st.number_input('Minimum number of iterations', min_value=0, max_value=1000, value=10,
                                 step=1)  # Wybór minimalnej liczby iteracji
        mi = st.number_input('Maximium number of iterations', min_value=0, max_value=1000, value=1000,
                                 step=1)  # Wybór maksymalnej liczby iteracji
        rs = st.number_input('Random state', min_value=0, max_value=10000, value=1234,
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