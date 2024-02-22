import dotenv
import os
import pandas as pd


from modules.dataset import DatasetLoader
from modules.analysis import FeatureAnalysis


reload_dataset = False
analyze_dataset = False
save_analysis = True
analysis_target = ""



if __name__ == "__main__":

    dotenv.load_dotenv()

    azure_connection_string = os.environ.get("AZURE_SQL_CONNECTION_STRING")

    list_of_tables_path = os.environ.get("LIST_OF_TABLE_NAMES")
    dataset_path = os.environ.get("DATASET_PATH")


    dataset = DatasetLoader(azure_connection_string, list_of_tables_path, dataset_path)


    if reload_dataset:
        dataset.load_and_save_dataset()

    if analyze_dataset:
        feature_analysis = FeatureAnalysis(dataset, analysis_target)

    dataset.load_and_print_tables_at_raw_data()





        
