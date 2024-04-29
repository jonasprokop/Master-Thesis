from modules.loader import Loader
from modules.analysis import DescriptiveAnalysis
from modules.dataset import DatasetMaker
from modules.models import Models
import pandas as pd



reload_dataset = False
remake_dataset = False
analyze_dataset = False
model = True


# Classes
# Registration

analysis_target = "Classes"
analysis_types = ["som-create-model", "som-plot-3d-scatter-category"]

#["k-means-elbow-method", "k-means-create-model", "k-means-silhouette-plot", "k-means-pca-scatter-plot", "aglomerative-hierarchical-clustering-silhouette-plot"]
#["aglomerative-hierarchical-clustering-create-model", "aglomerative-hierarchical-clustering-scatter-plot", "aglomerative-hierarchical-clustering-silhouette-plot"
# "aglomerative-hierarchical-clustering-summary", "aglomerative-hierarchical-clustering-plot-dendrogram"]
#["som-grid-search", "som-create-model", "som-summary" "som-silhouette-plot", "som-scatter-plot"]



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












        
