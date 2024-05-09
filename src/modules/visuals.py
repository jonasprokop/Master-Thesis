import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.colors
from sklearn.metrics import silhouette_samples, silhouette_score
from colormath.color_objects import HSVColor, sRGBColor
from colormath.color_conversions import convert_color
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import Counter


class Visuals():
    def _n_colors(self, n,method_offset):
        vals=[i/(n) for i in range(0,n)]
        colors=[]
        for val in vals:
            hsv_color = HSVColor(((val+method_offset)*360)%360,0.9,0.9)
            rgb_color = convert_color(hsv_color, sRGBColor)
            colors.append([rgb_color.rgb_r,rgb_color.rgb_g,rgb_color.rgb_b])
        arr=np.array(colors)
        return arr
    

    def _silhouette_plot_k_means(self):
        try:
            silhouette_avg = self._compute_silhouette_score(self._x_columns, self._y_km)
            sample_silhouette_values = silhouette_samples(self._x_columns, self._y_km)
        except:
            raise ValueError("K-means model must first be created")
        
        if not np.any(self._k_means_colors_list):
            max_y_km = max(self._y_km) + 1
            self._k_means_colors_list = np.array(self._n_colors(max_y_km,0.25))
        
        self._silhouette_plot(self._x_columns, self._k_means_n_clusters, self._y_km, 
                              self._k_means_colors_list, silhouette_avg, sample_silhouette_values)
        plt.show()

    def _silhouette_plot_hierarchical_clustering(self):
        try:
            silhouette_avg = self._compute_silhouette_score(self._x_columns, self._aglomerative_hierarchical_clusters)
            sample_silhouette_values = silhouette_samples(self._x_columns, self._aglomerative_hierarchical_clusters)
        except:
            raise ValueError("Hierarchical clustering model must first be created")

        if not np.any(self._aglomerative_hierarchical_colors_list):
            max_clusters = max(self._aglomerative_hierarchical_clusters) + 1
            self._aglomerative_hierarchical_colors_list = np.array(self._n_colors(max_clusters,0))
        
        self._silhouette_plot(self._x_columns, self._aglomerative_hierarchical_n_clusters, self._aglomerative_hierarchical_clusters, 
                              self._aglomerative_hierarchical_colors_list, silhouette_avg, sample_silhouette_values)
        plt.show()
        
    def _silhouette_plot_som(self):
        try:
            silhouette_avg = self._compute_silhouette_score(self._x_columns, self._som_clusters)
            sample_silhouette_values = silhouette_samples(self._x_columns, self._som_clusters)
        except:
            raise ValueError("SOM model must first be created")
        

        if not np.any(self._som_colors_list):
            max_clusters= max(self._som_clusters) + 1
            self._som_colors_list = np.array(self._n_colors(max_clusters,0.5))
        
        self._silhouette_plot(self._x_columns,  self._som_n_clusters,  self._som_clusters, 
                              self._som_colors_list, silhouette_avg, sample_silhouette_values)
    
        plt.show()

    def _silhouette_plot(self, pd_data, n_clusters, labels, colors_list, silhouette_avg, sample_silhouette_values):
        fig, ax1 = plt.subplots(1, 1)
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(pd_data) + (n_clusters+1) * 10])

        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, 
                            facecolor=colors_list[i], edgecolor=colors_list[i], alpha=0.7, label='Cluster '+str(i+1))
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(1+i))
            y_lower = y_upper + 10

        ax1.axvline(x=silhouette_avg, color="red", linestyle="--",label='Průměrné skóre')
        plt.legend()
        ax1.set_title("Graf silhouette skóre")
        ax1.set_xlabel("Hodnoty silhouette skóre")
        ax1.set_ylabel("Počet záznamů / Číslo clusteru")


    def _pca_scatter_plot_k_means(self):
        title = 'Zobrazení výsledků k-means clusteringu v první a druhé složce PCA'
        try:
            pca, x_pca = self._analysis._create_pca(self._x_columns, 2)
            transformed_centres = pca.transform(self._kmeans.cluster_centers_)
        except:
            raise ValueError("K-means model must first be created")

        max_clusters = max(self._y_km) + 1
        if not np.any(self._k_means_colors_list):
            self._k_means_colors_list = np.array(self._n_colors(max_clusters,0.25))

        self._pca_scatter_plot_prepare()
        for i in range(max_clusters):
            self._pca_scatter_plot_one_cluster(x_pca, self._y_km, self._k_means_colors_list, np.where(self._y_km==i),i)

        self._pca_scatter_plot(transformed_centres,title)

    def _pca_scatter_plot_hierarchical_clustering(self):
        title = 'Zobrazení výsledků hierarchického clusteringu v první a druhé složce PCA'
        try:
            pca, x_pca = self._analysis._create_pca(self._x_columns, 2)
            centroids = []
            for cluster_id in range(len(self._aglomerative_hierarchical_clusters) + 1):
                centroid = np.mean(x_pca[self._aglomerative_hierarchical_clusters == cluster_id], axis=0)
                centroids.append(centroid)
            centroids = np.array(centroids)
        except:
            raise ValueError("Hierarchical model must first be created")

        max_clusters = max(self._aglomerative_hierarchical_clusters) + 1

        if not np.any(self._aglomerative_hierarchical_colors_list):
            self._aglomerative_hierarchical_colors_list = np.array(self._n_colors(max_clusters,0))

        self._pca_scatter_plot_prepare()
        
        for i in range(max_clusters):
            self._pca_scatter_plot_one_cluster(x_pca, self._aglomerative_hierarchical_clusters, 
                                               self._aglomerative_hierarchical_colors_list, np.where(self._aglomerative_hierarchical_clusters==i),i)

        self._pca_scatter_plot(centroids,title)

    def _pca_scatter_plot_som(self):
        title = 'Zobrazení výsledků SOM v první a druhé složce PCA'
        try:
            pca, x_pca = self._analysis._create_pca(self._x_columns, 2)
            centroids = []
            for cluster_id in range(len(self._som_clusters) + 1):
                centroid = np.mean(x_pca[self._som_clusters == cluster_id], axis=0)
                centroids.append(centroid)
            centroids = np.array(centroids)
        except:
            raise ValueError("Som model must first be created")

        max_clusters = max(self._som_clusters) + 1

        if not np.any(self._som_colors_list):
            self._som_colors_list = np.array(self._n_colors(max_clusters,0.5))

        self._pca_scatter_plot_prepare()
        
        for i in range(max_clusters):
            self._pca_scatter_plot_one_cluster(x_pca, self._som_clusters, 
                                               self._som_colors_list, np.array(np.where(self._som_clusters==i)),i)
            
        self._pca_scatter_plot(centroids,title)

    def _pca_scatter_plot_prepare(self):
        plt.figure(figsize=(10, 6))

    def _pca_scatter_plot_one_cluster(self, x_pca, labels, colors_list, indicies, index):

        if len(indicies)==1: 
            indicies = indicies[0]

        plt.scatter(x_pca[indicies, 0], x_pca[indicies, 1], c=colors_list[labels[indicies]], s=50,label='Cluster '+str(index+1))

    def _pca_scatter_plot(self, transformed_centres, title):
        plt.scatter(transformed_centres[:, 0], transformed_centres[:, 1], s=75, marker='x', c='black', label='Centroidy')
        plt.title(title)
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def _create_clusters_summary_k_means(self, columns_select, rename_dict, head, days_in_week):
        pd_data_not_recoded = self._pd_data_not_recoded
        pd_data_not_recoded["Cluster"] = self._y_km
        columns_select.append("Cluster")
        pd_data_not_recoded = pd_data_not_recoded[columns_select]
        pd_data_not_recoded_renamed = pd_data_not_recoded.rename(columns=rename_dict)
        pd_data_not_recoded_renamed["Den"] = pd_data_not_recoded_renamed["Den"].apply(lambda x: self._keep_only_day(x, days_in_week))
        pd_data_cluster_summary = self._summarize_clusters(pd_data_not_recoded_renamed, "Cluster", head)
        return pd_data_cluster_summary
    
    def _create_clusters_summary_hierarchical_clustering(self, columns_select, rename_dict, head, days_in_week):
        pd_data_not_recoded = self._pd_data_not_recoded
        pd_data_not_recoded["Cluster"] = self._aglomerative_hierarchical_clusters 
        columns_select.append("Cluster")
        pd_data_not_recoded = pd_data_not_recoded[columns_select]
        pd_data_not_recoded_renamed = pd_data_not_recoded.rename(columns=rename_dict)
        pd_data_not_recoded_renamed["Den"] = pd_data_not_recoded_renamed["Den"].apply(lambda x: self._keep_only_day(x, days_in_week))
        pd_data_cluster_summary = self._summarize_clusters(pd_data_not_recoded_renamed, "Cluster", head)
        return pd_data_cluster_summary
    
    def _create_clusters_summary_som(self, columns_select, rename_dict, head, days_in_week):
        pd_data_not_recoded = self._pd_data_not_recoded
        pd_data_not_recoded["Cluster"] = self._som_clusters
        columns_select.append("Cluster")
        pd_data_not_recoded = pd_data_not_recoded[columns_select]
        pd_data_not_recoded_renamed = pd_data_not_recoded.rename(columns=rename_dict)
        pd_data_not_recoded_renamed["Den"] = pd_data_not_recoded_renamed["Den"].apply(lambda x: self._keep_only_day(x, days_in_week))
        pd_data_cluster_summary = self._summarize_clusters(pd_data_not_recoded_renamed, "Cluster", head)
        return pd_data_cluster_summary
            
    def _summarize_clusters(self, pd_data, cluster_column, head):
        grouped = pd_data.groupby(cluster_column)
        summaries = []
        
        for cluster, group in grouped:
            cluster_summary = self._create_cluster_summary(cluster, group, cluster_column, head)
            summaries.append(cluster_summary)
        summary_pd_data = pd.DataFrame(summaries)
        
        return summary_pd_data
    
    def _create_cluster_summary(self, cluster, group, cluster_column, head):
        kapacita = None
        cluster_summary = {'Cluster': cluster}

        cluster_length = len(group)
        cluster_summary['Počet záznamů'] = cluster_length
        
        for column in group.columns:
            if column == cluster_column:
                continue

            elif column == "Kapacita":
                kapacita = group[column].mean()
                mean_value = round(kapacita, 2)
                cluster_summary[column] = mean_value

            elif column == "Neobsazeno":
                neobsazeno = group[column].mean()
                mean_value = round(neobsazeno, 2)
                pomer = (neobsazeno / kapacita)*100
                cluster_summary[column] = mean_value
                cluster_summary["Poměrná neobsazenost"] = str(round(pomer, 0)) + " %"

            elif pd.api.types.is_numeric_dtype(group[column]):
                mean_value = group[column].mean()
                mean_value = round(mean_value, 2)
                cluster_summary[column] = mean_value
            else:
                value_counts = group[column].value_counts().head(head).index.tolist()
                value_counts = [str(x) for x in value_counts if x not in [None, '-']]
                cluster_summary[column] = ", ".join(value_counts)

        return cluster_summary
                    
    def _keep_only_day(self, day_str, days_in_week):
        if day_str in days_in_week:
            return day_str
        else:
            return "bez opakování"
        
    def _plot_dendrogram_hierarchical_clustering(self, pd_data, n_clusters, model, p, cluster_labels, n_colors, label_divisor, x_font_size):

        linkage_matrix = linkage(pd_data, model.linkage)
        distances = model.distances_
        plt.figure(figsize=(12, 8))  

        def plot_dendrogram_with_clusters(model, pd_data, link_m, **kwargs):
            dendrogram(link_m, **kwargs)
        
        sorted_distances = np.sort(distances)[::-1]
        c_treshold = sorted_distances[n_clusters-2]
        color_list=[matplotlib.colors.to_hex(color) for color in n_colors[cluster_labels]]

        plt.yscale('log')
        x_var=[0]

        print(len(linkage_matrix))
        link_cols = {}
        for i, i12 in enumerate(linkage_matrix[:, :2].astype(int)):
            c1, c2 = (link_cols[x] if x > len(linkage_matrix) else color_list[x]
                        for x in i12)
            link_cols[i + 1 + len(linkage_matrix)] = c1 if c1 == c2 else 'C0'
        
        plot_dendrogram_with_clusters(model, pd_data, linkage_matrix, truncate_mode='level', p = p, 
                                        color_threshold = c_treshold, leaf_label_func= lambda id : self._implemented_leaf_label_function(id, x_var, label_divisor),
                                        leaf_font_size=x_font_size,leaf_rotation=80,link_color_func=lambda x:link_cols[x])

        
        plt.legend(handles=[plt.Line2D([],[],color=color,label="Cluster "+str(label)) for color,label in zip(n_colors[0:n_clusters],range(1,1+n_clusters))],bbox_to_anchor=(1.1,1.15),framealpha=1)

        plt.title('Dendrogram hierarchického clusteringu s barevně odlišenými nalezenými clustery')
        plt.xlabel('Index')
        plt.ylabel('Vzdálenost')
        plt.show()

    def _implemented_leaf_label_function(self, id,x_var, label_divisor):
        x_var[0]=x_var[0]+1
        if x_var[0]%label_divisor==0:
            return str(id)
        else:
            return ""

    def _count_pca_component_division(self, pd_data):
        n_components = 2
        pca, pca_pd_data = self._analysis._create_pca(pd_data, n_components)
        explained_variances = pca.explained_variance_ratio_
        explained_variance_ratio = explained_variances[0] / explained_variances[1]
        return explained_variance_ratio

    def _plot_3d_scatter_categories(self, som, pd_data_not_recoded, pd_data, category):
        values = pd_data.values
        unique_categories_dict=Counter(pd_data_not_recoded[category])
        sorted_unique_categories_dict = dict(sorted(unique_categories_dict.items(), key=lambda x: x[1], reverse=True))
        unique_vals=list(sorted_unique_categories_dict.keys())

        def get_index(categ):
            return unique_vals.index(categ)
        new_data_sorted=pd_data_not_recoded.sort_values(by=category,key=lambda x:x.map(get_index))
        color_vals= self._n_colors(len(unique_vals),0.0)
        
        plt.figure(figsize=(12, 8))
        ax = plt.axes(projection='3d')

        vals=[]
        
        for i in range(len(new_data_sorted)):
            alocced_val=new_data_sorted.iloc[i]
            index=alocced_val.name
            color_index=unique_vals.index(alocced_val[category])
            x=values[index]
            w = som.winner(x)
            vals.append((w[0],w[1],i,color_index))

        points_x=[point[0] for point in vals]
        points_y=[point[1] for point in vals]
        points_z=[point[2] for point in vals]
        points_c=[color_vals[point[3]] for point in vals]

        ax.scatter(points_x, points_y, points_z, c=points_c, marker='o')

        ax.set_xlabel('Mapa osa x')
        ax.set_ylabel('Mapa osa y')
        ax.set_zlabel('Index datového bodu')
        ax.set_title('3D bodový graf rozdělení datových bodů do clusterů')

        max_legend=10


        plt.legend(handles=[plt.Line2D([], [], color=color, label=label) for color, 
                            label in zip(color_vals[0:max_legend][::-1], unique_vals[0:max_legend][::-1])])

        plt.show()



  





        

