from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import DBSCAN 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from itertools import product
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances

from modules.visuals import Visuals
from minisom import MiniSom
from collections import Counter


class Models(Visuals):
    def __init__(self,
            analysis):
        

        self._analysis = analysis
        self._analysis_target = analysis._analysis_target
        self._dataset = analysis._dataset
        self._pd_data = analysis._pd_data
        self._pd_data_not_recoded = analysis._pd_data_not_recoded
        self._x_config = analysis._x_config
        self._y_config = analysis._y_config
        self._desc_config = analysis._desc_config
        self._x_columns = analysis._x_columns
        self._y_columns = analysis._y_columns
        self._desc_columns = analysis._desc_columns
        self._config = analysis._dataset._loader._model_data[analysis._analysis_target]

        self._kmeans = None
        self._y_km = None
        self._k_means_n_clusters = None
        self._k_means_colors_list = None

        self._aglomerative_hierarchical_n_clusters = None
        self._aglomerative_hierarchical_model = None
        self._aglomerative_hierarchical_clusters = None
        self._aglomerative_hierarchical_colors_list = None

        self._som_model =  None 
        self._som_clusters = None
        self._som_n_clusters = None
        self._som_colors_list = None

        self._dbscan = None
        self._clusters_dbscan = None

        

    def create_analysis(self, analysis_types):
        for analysis_type in analysis_types:
            if analysis_type == "k-means-grid-search":
                config = self._analysis._dataset._load_variable_from(self._config, "k_means_grid_search")
                n_clusters = self._analysis._dataset._load_variable_from(config, "n_clusters")
                init = self._analysis._dataset._load_variable_from(config, "init")
                n_init = self._analysis._dataset._load_variable_from(config, "n_init")
                max_iter = self._analysis._dataset._load_variable_from(config, "max_iter")
                tol = self._analysis._dataset._load_variable_from(config, "tol")
                grid_search = self._calculate_best_k_means_params(self._x_columns, 
                                                    n_clusters, init, n_init, max_iter, tol)

                print("Best parameters found: ", grid_search.best_params_)
                print("Best score: ", grid_search.best_score_)

            if analysis_type == "k-means-elbow-method":
                config =  self._analysis._dataset._load_variable_from(self._config, "k_means_elbow_method")
                init = self._analysis._dataset._load_variable_from(config, "init")
                n_init = self._analysis._dataset._load_variable_from(config, "n_init")
                max_iter = self._analysis._dataset._load_variable_from(config, "max_iter")
                tol = self._analysis._dataset._load_variable_from(config, "tol")
                random_state = self._analysis._dataset._load_variable_from(config, "random_state")
                range_interval = self._analysis._dataset._load_variable_from(config, "range_interval")
                self._elbow_method(self._x_columns, init, n_init, max_iter, tol, random_state, range_interval)

            if analysis_type == "k-means-create-model":
                config = self._analysis._dataset._load_variable_from(self._config, "k_means_create_model")
                n_clusters = self._analysis._dataset._load_variable_from(config, "n_clusters")
                init = self._analysis._dataset._load_variable_from(config, "init")
                n_init = self._analysis._dataset._load_variable_from(config, "n_init")
                max_iter = self._analysis._dataset._load_variable_from(config, "max_iter")
                tol = self._analysis._dataset._load_variable_from(config, "tol")
                random_state = self._analysis._dataset._load_variable_from(config, "random_state")
                self._fit_predict_k_means(self._x_columns, n_clusters, init, n_init, max_iter, tol, random_state)

            if analysis_type == "k-means-silhouette-plot":
                self._silhouette_plot_k_means()

            if analysis_type == "k-means-pca-scatter-plot":
                self._pca_scatter_plot_k_means()

            if analysis_type == "k-means-summary":
                config = self._analysis._dataset._load_variable_from(self._config, "k_means_cluster_summary")
                columns_select = self._analysis._dataset._load_variable_from(config, "columns_select")
                rename_dict = self._analysis._dataset._load_variable_from(config, "rename_dict") 
                head = self._analysis._dataset._load_variable_from(config, "head") 
                days_in_week = self._analysis._dataset._load_variable_from(config, "days_in_week")
                pd_data_cluster_summary = self._create_clusters_summary_k_means(columns_select, rename_dict, head, days_in_week)
                path_cluster_summary = f"C://TEMP/{self._analysis_target}_k_means_cluster_summary.xlsx"
                pd_data_cluster_summary.to_excel(path_cluster_summary)

            if analysis_type == "aglomerative-hierarchical-clustering-grid-search":
                config = self._analysis._dataset._load_variable_from(self._config, "aglomerative_hierarchical_clustering_grid_search")
                n_clusters = self._analysis._dataset._load_variable_from(config, "n_clusters")
                linkage = self._analysis._dataset._load_variable_from(config, "linkage")
                best_model, pd_data_hierarchical_grid_search = self._fit_predict_aglomerative_hierarchical_clustering_grid_search(self._x_columns, linkage, 
                                                                                                 n_clusters)
                print(best_model, pd_data_hierarchical_grid_search)
            if analysis_type == "aglomerative-hierarchical-clustering-create-model":
                config = self._analysis._dataset._load_variable_from(self._config, "aglomerative_hierarchical_clustering_create_model")
                n_clusters = self._analysis._dataset._load_variable_from(config, "n_clusters")
                linkage = self._analysis._dataset._load_variable_from(config, "linkage")
                self._fit_predict_aglomerative_hierarchical_clustering(self._x_columns, n_clusters, linkage)


            if analysis_type == "aglomerative-hierarchical-clustering-plot-dendrogram":
                config = self._analysis._dataset._load_variable_from(self._config, "aglomerative_hierarchical_clustering_plot_dendrogram")
                p = self._analysis._dataset._load_variable_from(config, "p")
                self._aglomerative_hierarchical_colors_list=self._n_colors(self._aglomerative_hierarchical_n_clusters,0)
                self._plot_dendrogram_hierarchical_clustering(self._x_columns, 
                                                              self._aglomerative_hierarchical_n_clusters, self._aglomerative_hierarchical_model, p,self._aglomerative_hierarchical_clusters,self._aglomerative_hierarchical_colors_list,50,9)
            if analysis_type == "aglomerative-hierarchical-clustering-silhouette-plot":    
                self._silhouette_plot_hierarchical_clustering()

            if analysis_type == "aglomerative-hierarchical-clustering-scatter-plot":    
                self._pca_scatter_plot_hierarchical_clustering()

            if analysis_type == "aglomerative-hierarchical-clustering-summary":
                config = self._analysis._dataset._load_variable_from(self._config, "aglomerative_hierarchical_clustering_summary")
                columns_select = self._analysis._dataset._load_variable_from(config, "columns_select")
                rename_dict = self._analysis._dataset._load_variable_from(config, "rename_dict") 
                head = self._analysis._dataset._load_variable_from(config, "head") 
                days_in_week = self._analysis._dataset._load_variable_from(config, "days_in_week")
                pd_data_cluster_summary = self._create_clusters_summary_hierarchical_clustering(columns_select, rename_dict, head, days_in_week)
                path_cluster_summary = f"C://TEMP/{self._analysis_target}_aglomerative_hierarchical_clustering_summary.xlsx"
                pd_data_cluster_summary.to_excel(path_cluster_summary)

            if analysis_type == "som-grid-search":
                config = self._analysis._dataset._load_variable_from(self._config, "som_grid_search")
                ms = self._analysis._dataset._load_variable_from(config, "ms")
                ns = self._analysis._dataset._load_variable_from(config, "ns")
                train_counts = self._analysis._dataset._load_variable_from(config, "train_counts")
                self._grid_search_parametrs_som(ms, ns, train_counts, self._x_columns)

            if analysis_type == "som-create-model":
                config = self._analysis._dataset._load_variable_from(self._config, "som_create_model")
                m = self._analysis._dataset._load_variable_from(config, "m")
                n = self._analysis._dataset._load_variable_from(config, "n")
                train_count = self._analysis._dataset._load_variable_from(config, "train_count")
                self._fit_predict_som(self._x_columns, m, n, train_count) 
            
            if analysis_type == "som-summary":
                config = self._analysis._dataset._load_variable_from(self._config, "som_summary")
                columns_select = self._analysis._dataset._load_variable_from(config, "columns_select")
                rename_dict = self._analysis._dataset._load_variable_from(config, "rename_dict") 
                head = self._analysis._dataset._load_variable_from(config, "head") 
                days_in_week = self._analysis._dataset._load_variable_from(config, "days_in_week")
                pd_data_cluster_summary = self._create_clusters_summary_som(columns_select, rename_dict, head, days_in_week)
                path_cluster_summary = f"C://TEMP/{self._analysis_target}_som_summary.xlsx"
                pd_data_cluster_summary.to_excel(path_cluster_summary)

            if analysis_type == "som-silhouette-plot":    
                self._silhouette_plot_som()

            if analysis_type == "som-scatter-plot":    
                self._pca_scatter_plot_som()

            if analysis_type == "som-plot-3d-scatter-category":
                config = self._analysis._dataset._load_variable_from(self._config, "som_3d_scatter")
                category = self._analysis._dataset._load_variable_from(config, "category")
                self._plot_3d_scatter_categories(self._som_model,
                                                self._pd_data_not_recoded, self._x_columns, category)
                

    def _calculate_best_k_means_params(self, pd_data, n_clusters, init, n_init, max_iter, tol): 
        param_grid = {
            'n_clusters':n_clusters, 
            'init':init,  
            'n_init':n_init, 
            'max_iter':max_iter,
            'tol': tol 
        }

        kmeans = KMeans()

        grid_search = GridSearchCV(kmeans, param_grid, cv=5)

        grid_search.fit(pd_data)

        return grid_search

    def _fit_predict_k_means(self, pd_data, n_clusters, init, n_init, max_iter, tol, random_state):
        kmeans = KMeans(
        n_clusters=n_clusters, init=init,
        n_init=n_init, max_iter=max_iter,
        tol=tol, random_state=random_state
        )

        y_km = kmeans.fit_predict(pd_data)

        self._kmeans = kmeans
        self._y_km = y_km
        self._k_means_n_clusters = n_clusters


    
    def _elbow_method(self, pd_data, init, n_init, max_iter, tol, random_state, range_interval):
        distortions = []
        for i in range(range_interval[0], range_interval[1]):
            km = KMeans(
                    n_clusters=i, init=init,
                    n_init=n_init, max_iter=max_iter,
                    tol=tol, random_state=random_state
                )
            km.fit(pd_data)
            distortions.append(km.inertia_)

        plt.plot(range(range_interval[0], range_interval[1]), distortions, marker='o')
        plt.title("Hodnota deformace při daném počtu clusterů")
        plt.xlabel('Počet clusterů')
        plt.ylabel('Deformace')
        plt.show()


    def _fit_predict_aglomerative_hierarchical_clustering_grid_search(self, pd_data, linkage_options, n_clusters_options):
        best_score = -1
        best_model = None
        results_list = []
        index = 0
        threshold = 100

        for linkage in linkage_options:
            for n_clusters in n_clusters_options:
                index += 1
                model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
                clusters = model.fit_predict(pd_data)
                silhouette_avg = self._compute_silhouette_score(pd_data, clusters)
                wcss = self. _compute_wcss(pd_data, clusters, n_clusters)
                results_list.append({"linkage": linkage, "n_clusters": n_clusters, "silhouette_avg": silhouette_avg, "wcss":wcss})
                
                if silhouette_avg > best_score and wcss > threshold:
                    best_score = silhouette_avg
                    best_model = model

        pd_data_hierarchical_grid_search = pd.DataFrame(results_list)
        return best_model, pd_data_hierarchical_grid_search
    
    def _compute_silhouette_score(self, pd_data, clusters):
        silhouette_avg = silhouette_score(pd_data, clusters)
        return silhouette_avg

    def _compute_wcss(self, pd_data, clusters, n_clusters):
        wcss = 0
        for cluster_idx in range(n_clusters):
            cluster_points = pd_data[clusters == cluster_idx]
            cluster_center = np.mean(cluster_points, axis=0)
            wcss += np.sum(pairwise_distances(cluster_points, [cluster_center])**2)
        return wcss
    
    def _fit_predict_aglomerative_hierarchical_clustering(self, pd_data, n_clusters, linkage):
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, compute_distances=True)
        clusters = model.fit_predict(pd_data)
        self._aglomerative_hierarchical_n_clusters = n_clusters
        self._aglomerative_hierarchical_model = model
        self._aglomerative_hierarchical_clusters = clusters
    
    def _grid_search_parametrs_som(self, ms, ns, train_counts, pd_data):
        values = pd_data.values
        scores_list = []
        best_silhouette_score = -1 
        permitted_num_of_clusters = 12
        best_model_definiton = None
        
        for m in ms:
            for n in ns:
                for train_count in train_counts:
                    som = MiniSom(m, n, values.shape[1], sigma=0.5, learning_rate=0.5, 
                                neighborhood_function='gaussian', random_seed=10)
                    som.train_random(values, train_count)  
                    
                    bmu_indices = np.array([som.winner(x) for x in values])
                    
                    num_neurons = som.get_weights().shape[0] * som.get_weights().shape[1]
                    cluster_labels = np.ravel_multi_index(bmu_indices.T, (m, n))
                    number_of_clusters = len(np.unique(cluster_labels))
                    
                    silhouette = silhouette_score(values, cluster_labels)
                    result_dict = {"m-n":str(m) + "-" + str(n),"train_count":train_count, 
                                        "len_cluster_labels":number_of_clusters,"silhouette score":silhouette}
                    scores_list.append(result_dict)
                
                    if silhouette > best_silhouette_score and number_of_clusters < permitted_num_of_clusters:
                        best_silhouette_score = silhouette
                        best_model_definiton = result_dict
                pd_data_scores = pd.DataFrame(scores_list)     

        return best_model_definiton, pd_data_scores
    
    def _fit_predict_som(self, pd_data, m, n, train_count):
        values = pd_data.values

        som = MiniSom(m, n, values.shape[1], sigma=0.5, learning_rate=0.5, 
            neighborhood_function='gaussian', random_seed=10)
        som.train_random(values, train_count)

        self._som_model =  som 
        self._som_clusters = self._join_cluster_columns(pd_data, som)
        self._som_n_clusters = m*n
        
    def _grid_search_dbscan(eps_values, pd_data, min_samples_values, worst_permitted_div_of_outliers, min_labels, max_labels): 
        best_score = -1
        best_percentage_of_outliers = 100
        best_params = {}
        combination_number = 0
        percentage_dict = {}

        for eps, min_samples in product(eps_values, min_samples_values):
            combination_number +=1 
            print(f"The combination is eps:{eps}; min_samples:{min_samples}")
            db_default = DBSCAN(eps = eps, min_samples = min_samples).fit(pd_data) 
            labels = db_default.labels_ 
        
            num_outliers = sum(labels == -1)
            len_labels = len(set(labels))
            num_of_points = len(pd_data)
            percentage_of_outliers = (num_outliers/num_of_points)*100
            percentage_dict.update({combination_number:[percentage_of_outliers, len_labels]})
        
            if len_labels >= min_labels and len_labels < max_labels and num_outliers < (num_of_points)/worst_permitted_div_of_outliers:

                score = silhouette_score(pd_data, labels)
                if score > best_score and percentage_of_outliers < best_percentage_of_outliers:
                    best_score = score
                    best_percentage_of_outliers = percentage_of_outliers
                    best_params = {'eps': eps, 'min_samples': min_samples, "percentage_of_outliers":percentage_of_outliers}

            return best_params
        
    def _fit_predict_dbscan(self, pd_data, eps, min_samples):
        dbscan = DBSCAN(eps = eps, min_samples = min_samples).fit(pd_data) 
        clusters = dbscan.labels_ 
        self._dbscan = dbscan
        self._clusters_dbscan = clusters


    def _get_cluster_values(self, som, pd_data):
        cluster_values = []
        cluster_counter = Counter()  
        values = pd_data.values

        for x in values:
            w = som.winner(x)
            cluster_values.append(w)
            cluster_counter[w] += 1  
            
        return cluster_values, cluster_counter
    
    def _join_cluster_columns(self, pd_data, som):
        cluster_values, cluster_counter = self._get_cluster_values(som, pd_data)
        pd_data_clusters = pd.DataFrame(cluster_values, columns=['Cluster_X', 'Cluster_Y'])
        pd_data_clusters["Cluster"] = pd_data_clusters['Cluster_X'].astype(str) + pd_data_clusters['Cluster_Y'].astype(str)
        cluster_mapping = {cluster: idx for idx, cluster in enumerate(pd_data_clusters["Cluster"].unique())}
        pd_data_clusters["Cluster"] = pd_data_clusters["Cluster"].map(cluster_mapping)
        return pd_data_clusters["Cluster"]

