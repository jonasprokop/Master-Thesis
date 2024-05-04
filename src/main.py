from modules.loader import Loader
from modules.analysis import DescriptiveAnalysis
from modules.dataset import DatasetMaker
from modules.models import Models
import pandas as pd

reload_dataset = False
remake_dataset = False
analyze_dataset = False
model = True

analysis_target = "Classes"
analysis_types = ["som-create-model", "som-plot-3d-scatter-category"]


if __name__ == "__main__":

    loader = Loader()

    if reload_dataset:
        loader.load_and_save_dataset()

    dataset = DatasetMaker(loader)

    if remake_dataset:
        dataset.create_datasets()

    descriptive_analysis = DescriptiveAnalysis(dataset, analysis_target)

    if analyze_dataset:
        descriptive_analysis.analyse_dataset()

        
    models = Models(descriptive_analysis)

    if model:
        models.create_analysis(analysis_types)












        
