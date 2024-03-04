import pandas as pd

from modules.loader import DatasetLoader
from modules.analysis import FeatureAnalysis
from modules.dataset import DatasetMaker


reload_dataset = False
remake_dataset = True
analyze_dataset = False
save_analysis = False
analysis_target = ""



if __name__ == "__main__":

    dataset_loader = DatasetLoader()

    if reload_dataset:
        dataset_loader.load_and_save_dataset()

    dataset = DatasetMaker(dataset_loader)

    if remake_dataset:
        dataset.create_test_dataset()

    if analyze_dataset:
        feature_analysis = FeatureAnalysis(dataset, analysis_target)






        
