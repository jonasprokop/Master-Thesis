import dotenv
import os
import pandas as pd
import json
import yaml
import unicodedata
from connections.database_conn import AzureLoader


class Loader():
    def __init__(self):
        
        dotenv.load_dotenv()

        azure_connection_string = os.environ.get("AZURE_SQL_CONNECTION_STRING")
        model_metadata = os.environ.get("MODEL_METADATA")
        dataset_path = os.environ.get("DATASET_PATH")
        mapper = os.environ.get("MAPPER")
        subjects_dataset_tables = os.environ.get("SUBJECTS_DATASET_TABLES")
        subjects_dataset_operations = os.environ.get("SUBJECTS_DATASET_OPERATIONS")
        excel_data = os.environ.get("EXCEL_DATA")

        self._azure_loader = AzureLoader(azure_connection_string)
        self._dataset_path = dataset_path
        self._model_metadata = self._load_json(model_metadata)
        self._mapper = self._load_json(mapper)
        self._subjects_dataset_tables = self._load_json(subjects_dataset_tables)
        self._subjects_dataset_operations = self._load_yaml(subjects_dataset_operations)
        self._excel_data = excel_data


    def load_and_save_dataset(self):
        for table in self._model_metadata:
            database_table_name = self._mapper[table]
            query = f"SELECT * FROM {database_table_name}"

            pd_data = self._azure_loader.fetch_from_database(query)

            print(f"Table {table} was downloaded from source db")

            pd_data.columns = [self._strip_accents(column) for column in pd_data.columns]

            self.save_raw_table(self, table, pd_data)

    def load_raw_table(self, table):
        if table in self._model_metadata:
            path = self._dataset_path + "/raw-tables/" + table 
            pd_data = self._load_table_from_parquet(path)

            print(f"Table {table} was loaded from cache")
            return pd_data
        
    def save_raw_table(self, table, pd_data):
        path = self._dataset_path + "/raw-tables/*" + table 
        self._save_table_to_parquet(path, pd_data)
        
        print(f"Table {table} was saved into parquet cache")
    
    def save_table_for_analysis(self, table, pd_data):
        path = self._dataset_path + "/analysis-tables/" + table
        self._save_table_to_parquet(path, pd_data)
        
        print(f"Table {table} was saved into parquet cache")
        
    def load_table_for_analysis(self, table):
        path = self._dataset_path + "/analysis-tables/" + table 
        pd_data = self._load_table_from_parquet(path)

        print(f"Table {table} was loaded from cache")
        return pd_data

    def _load_table_from_parquet(self, path):
        pd_data = pd.read_parquet(path)
        return pd_data

    def _save_table_to_parquet(self, path, pd_data):
            pd_data.to_parquet(path)

    def _strip_accents(self, text):
        return ''.join(c for c in unicodedata.normalize('NFKD', text) if unicodedata.category(c) != 'Mn')
    
    def _load_json(self, path):
        with open(path, 'r', encoding="utf-8") as json_file:
            loaded_json = json.load(json_file)
            return loaded_json

    def _load_yaml(self, path):
        with open(path, "r", encoding='utf-8') as yaml_file:
            yaml_content = yaml.safe_load(yaml_file)
            return yaml_content


    def _load_excel(self, path, sheet_name):
        pd_data = pd.read_excel(path, sheet_name=sheet_name)
        return pd_data

    
        
        