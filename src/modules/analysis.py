from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


from modules.loader import AzureLoader

class FeatureAnalysis():
    def __init__(self,
                loader,
                analysis_target
                ):
        
        self._loader = loader    
        self._analysis = analysis_target
    
    def select_k_best(self):
        print(x)
            
