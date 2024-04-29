import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from scipy.stats import pearsonr
import pandas as pd
import datetime
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import DBSCAN 
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import normalize 
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from itertools import product
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances


class DescriptiveAnalysis():
    def __init__(self,
                dataset,
                analysis_target
                ):
        
        analysis_target_final = analysis_target + "_final"
        analysis_target_not_recoded = analysis_target + "_not_recoded"
        
        try:
            pd_data = dataset._loader.load_table_for_analysis(analysis_target_final)
        except:
            ValueError("Table for analysis not yet created")

        try:
            pd_data_not_recoded = dataset._loader.load_table_for_analysis(analysis_target_not_recoded)
        except:
            ValueError("Table for analysis not yet created")


        config = dataset._loader._statistical_analysis[analysis_target]
        select_columns_config = dataset._load_variable_from(config, "variables") 
        x_config = dataset._load_variable_from(select_columns_config, "x_config") 
        y_config = dataset._load_variable_from(select_columns_config, "y_config") 
        desc_config = dataset._load_variable_from(select_columns_config, "desc_config") 

        x_columns, y_columns, desc_columns = self._select_columns_for_use(pd_data, x_config, y_config, desc_config)
        
        self._analysis_target = analysis_target
        self._dataset = dataset
        self._pd_data = pd_data
        self._pd_data_not_recoded = pd_data_not_recoded
        self._config = config
        self._x_config = x_config
        self._y_config = y_config
        self._desc_config = desc_config
        self._x_columns = x_columns
        self._y_columns = y_columns
        self._desc_columns = desc_columns

        self._fig_size = (10,6)

    def analyse_dataset(self):
        pca_config = self._dataset._load_variable_from(self._config, "pca_dim_reduction") 
        n_components = self._dataset._load_variable_from(pca_config, "n_components") 
        pca, pca_pd_data = self._create_pca(self._x_columns, n_components)

        colour = self._dataset._load_variable_from(pca_config, "colour") 
        s = self._dataset._load_variable_from(pca_config, "s") 
        self._plot_pca(pca_pd_data, colour, s)

        return self._column_summary(self._pd_data_not_recoded, self._x_config, self._y_config, self._desc_config)

    def _select_columns_for_use(self, pd_data, x_config, y_config, desc_config):
        x_columns = pd_data[x_config]
        y_columns = pd_data[y_config]
        desc_columns =pd_data[desc_config]
        return x_columns, y_columns, desc_columns


    def _create_pca(self, pd_data, n_components):
        pca = PCA(n_components=n_components)
        pca_pd_data = pca.fit_transform(pd_data)
        return pca, pca_pd_data

    def _plot_pca(self, pca_pd_data, colour, s):
        plt.figure(figsize=self._fig_size)
        plt.scatter(pca_pd_data[:, 0], pca_pd_data[:, 1], c=colour, s=s)
        plt.title('Bodový graf datových bodů v první a druhé hlavní komponentě získané z PCA')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.legend()
        plt.grid(True)
        plt.show()

    def _column_summary(self, pd_data, columns_list_x, columns_list_y, columns_list_desc):
        summaries = []
        for column in pd_data.columns:
            cluster_summary = {}
            skip_value = False
            if column in columns_list_x:
                type = "Nezávislá"
            elif column in columns_list_y:
                type = "Závislá"
            elif column in columns_list_desc:
                type = "Popisná"
            else:
                skip_value = True

            if not skip_value:

                cluster_summary["Jméno sloupce"] = column

                cluster_summary["Typ proměnné"] = type

                try:
                    pd_data[column] = pd_data[column].apply(int)
                    mean_value = round(pd_data[column].mean(), 0)
                    cluster_summary["Typ sloupce"] = "kontinuální"
                    cluster_summary["Střední hodnota"] = mean_value
                    cluster_summary["Nejčestnější hodnoty"] = "X"
                    isna = pd_data[column].isna().sum()
                    cluster_summary["Počet chybějících hodnot"] = isna
                except:
                    value_counts = pd_data[column].value_counts().head(3).index.tolist()
                    value_counts = [str(x) for x in value_counts if x not in [None, '-', 0, "?", "??"]]
                    top_values = ", ".join(value_counts)
                    cluster_summary["Typ sloupce"] = "kategorická"
                    cluster_summary["Nejčestnější hodnoty"] = top_values
                    cluster_summary["Střední hodnota"] = "X"
                    isna = pd_data[column].isna().sum()
                    cluster_summary["Počet chybějících hodnot"] = isna

                summaries.append(cluster_summary)

        summary_pd_data = pd.DataFrame(summaries)
        ordered_data = summary_pd_data.sort_values(by='Typ proměnné', ascending=False)
        path_analysis = f"C://TEMP/{self._analysis_target}_analysis_result.xlsx"
        ordered_data.to_excel(path_analysis)

