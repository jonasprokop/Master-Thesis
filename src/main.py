import pandas as pd

from modules.loader import Loader
from modules.analysis import FeatureAnalysis
from modules.dataset import DatasetMaker


reload_dataset = False
remake_dataset = True
analyze_dataset = False
save_analysis = False
analysis_target = ""

if __name__ == "__main__":

    loader = Loader()

    if reload_dataset:
        loader.load_and_save_dataset()

    if remake_dataset:
        dataset = DatasetMaker(loader)
        dataset.create_test_dataset()

    if analyze_dataset:
        feature_analysis = FeatureAnalysis(loader, analysis_target)






        
