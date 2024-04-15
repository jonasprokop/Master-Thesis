import pandas as pd

from modules.loader import Loader
from modules.analysis import FeatureAnalysis
from modules.dataset import DatasetMaker


reload_dataset = False
remake_dataset = True
analyze_dataset = True
save_analysis = False
analysis_target = "Classes_not_recoded"

if __name__ == "__main__":

    loader = Loader()

    if reload_dataset:
        loader.load_and_save_dataset()

    dataset = DatasetMaker(loader)

    if remake_dataset:
        dataset.create_datasets()

    if analyze_dataset:
        feature_analysis = FeatureAnalysis(loader, dataset, analysis_target)
        feature_analysis._analyse_dataset()








        
