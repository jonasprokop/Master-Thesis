from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


from modules.loader import AzureLoader

class FeatureAnalysis():
    def __init__(self,
                dataset,
                table_to_analyze
                ):
        
        self._table_to_analyze = table_to_analyze
        self._pd_data = dataset.load_raw_table_from_dataset(table_to_analyze)
        
        
    def select_k_best(self):
        # Should make a table with possible columns to be dependent variable, 
        # so probably make tables an csv with all the explicit info

        print(self._pd_data )

        for variable in self._pd_data.iterrows():
            print(x)

            
