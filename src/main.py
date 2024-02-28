import dotenv
import os
import pandas as pd


from modules.loader import DatasetLoader
from modules.analysis import FeatureAnalysis
from modules.dataset import DatasetMaker


reload_dataset = False
analyze_dataset = False
save_analysis = False
analysis_target = ""



if __name__ == "__main__":

    dotenv.load_dotenv()

    azure_connection_string = os.environ.get("AZURE_SQL_CONNECTION_STRING")

    list_of_tables_path = os.environ.get("LIST_OF_TABLE_NAMES")
    dataset_path = os.environ.get("DATASET_PATH")


    dataset_loader = DatasetLoader(azure_connection_string, list_of_tables_path, dataset_path)


    if reload_dataset:
        dataset_loader.load_and_save_dataset()

    dataset = DatasetMaker(dataset_loader)
    dataset.create_model_metadata()
    dataset.populate_model_with_data()

    if analyze_dataset:
        feature_analysis = FeatureAnalysis(dataset, analysis_target)






        
