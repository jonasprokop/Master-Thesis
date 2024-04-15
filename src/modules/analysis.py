import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

class FeatureAnalysis():
    def __init__(self,
                loader,
                dataset,
                analysis_target
                ):
        
        self._loader = loader    
        self._analysis_target = analysis_target
        self._dataset = dataset

    def _analyse_dataset(self):
        table_config = self._loader._statistical_analysis[self._analysis_target]
        pd_data = self._loader.load_table_for_analysis(self._analysis_target)
        variables = self._dataset._load_variable_from_(table_config, "variables")
        pd_data_variables = self._select_variables(pd_data, variables)
        pd_data_exploration = self._explore_data(pd_data_variables)
        self._save_exploration_data(pd_data_exploration)

    def _select_variables(self, pd_data, variables):
        pd_data_variables = pd_data[variables]
        return pd_data_variables

    def _explore_data(self, pd_data):
        numerical_results = []
        categorical_results = []
        frequencies_results = []
        proportions_results = []

        for column in pd_data.columns:
            if pd.api.types.is_numeric_dtype(pd_data[column]):
                mean = pd_data[column].mean()
                median = pd_data[column].median()
                mode = pd_data[column].mode()[0]
                std_dev = pd_data[column].std()
                minimum = pd_data[column].min()
                maximum = pd_data[column].max()
                percentile_25 = pd_data[column].quantile(0.25)
                percentile_50 = pd_data[column].quantile(0.5)
                percentile_75 = pd_data[column].quantile(0.75)
                missing_values = pd_data[column].isnull().sum()

                numerical_results.append({
                    'Variable': column,
                    'Type': 'Numerical',
                    'Mean': mean,
                    'Median': median,
                    'Mode': mode,
                    'Std Deviation': std_dev,
                    'Min': minimum,
                    'Max': maximum,
                    '25th Percentile': percentile_25,
                    '50th Percentile': percentile_50,
                    '75th Percentile': percentile_75,
                    'Missing Values': missing_values
                })

            else:
                mode = pd_data[column].mode()[0]
                missing_values = pd_data[column].isnull().sum()
                
                frequencies = pd_data[column].value_counts()
                
                for category, frequency in frequencies.items():
                    frequencies_results.append({
                        'Variable': f"{column}_{category}",
                        'Frequencies': frequency
                    })

                proportions = frequencies / len(pd_data[column])

                for category, proportion in proportions.items():
                    proportions_results.append({
                        'Variable': f"{column}_{category}",
                        'Frequencies': proportion
                    })


                categorical_results.append({
                    'Variable': column,
                    'Type': 'Categorical',
                    'Mode': mode,
                    'Missing Values': missing_values
                })

        pd_data_numerical_results = pd.DataFrame(numerical_results)
        pd_data_categorical_results = pd.DataFrame(categorical_results)
        pd_data_frequencies_results = pd.DataFrame(frequencies_results)
        pd_data_proportions_results = pd.DataFrame(proportions_results)

        return [pd_data_numerical_results, pd_data_categorical_results, pd_data_frequencies_results, pd_data_proportions_results]
    

    def _save_exploration_data(self, pd_data_exploration):

        index = 0
        names_dict = {
                    0:"numerical_results",
                    1:"categorical_results",
                    2:"frequencies_results",
                    3:"proportions_results"
            }

        for table in pd_data_exploration:
            table.to_excel(f"C:/TEMP/{self._analysis_target + names_dict[index]}_excel_statistics.xlsx")
            index += 1


    
    def _visualize_correlation_matrix(self, correlation_matrix):
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Cross-korelační tabulka')
        plt.show()

