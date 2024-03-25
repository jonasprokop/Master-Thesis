import dotenv
import os
import pandas as pd
import json
import yaml
import unicodedata
import openpyxl



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
        classes_dataset_tables = os.environ.get("CLASSES_DATASET_TABLES")
        classes_dataset_operations = os.environ.get("CLASSES_DATASET_OPERATIONS")
        registration_dataset_tables = os.environ.get("REGISTRATION_DATASET_TABLES")
        registration_dataset_operations = os.environ.get("REGISTRATION_DATASET_OPERATIONS")

        self._azure_loader = AzureLoader(azure_connection_string)
        self._dataset_path = dataset_path
        self._model_metadata = self._load_json(model_metadata)
        self._mapper = self._load_json(mapper)
        self._subjects_dataset_tables = self._load_json(subjects_dataset_tables)
        self._subjects_dataset_operations = self._load_yaml(subjects_dataset_operations)
        self._excel_data = self._load_json(excel_data)
        self._classes_dataset_tables = self._load_json(classes_dataset_tables)
        self._classes_dataset_operations = self._load_yaml(classes_dataset_operations)
        self._registration_dataset_tables = self._load_json(registration_dataset_tables)
        self._registration_dataset_operations = self._load_yaml(registration_dataset_operations)


    def load_and_save_dataset(self):
        self._save_tables_from_db(self._mapper)
        self._save_tables_from_excel(self._excel_data)

    
    def _save_tables_from_db(self, mapper): 
        for table, database_table_name  in mapper.items():
            query = f"SELECT * FROM {database_table_name}"

            pd_data = self._azure_loader.fetch_from_database(query)

            print(f"Table {table} was downloaded from source db")

            pd_data.columns = [self._strip_accents(column) for column in pd_data.columns]

            self._save_raw_table(self, table, pd_data)

    def _save_tables_from_excel(self, excel_config):
         for table, table_data in excel_config.items():

            partial_table_paths = table_data["data"]
            skiprows = table_data["skiprows"]

            partial_tables = []
            pd_data = pd.DataFrame

            for partial_table_path in partial_table_paths:
                partial_table = self._load_excel(partial_table_path, sheet_name="Sheet1", skiprows=skiprows)
                partial_tables.append(partial_table)
            
            if partial_tables:
                pd_data = pd.concat(partial_tables, axis=0, ignore_index=True)
                pd_data.columns = [self._strip_accents(column) for column in pd_data.columns]
                print(f"I have succesfully concatened {table}")
                self._save_raw_table(table, pd_data)

    def load_raw_table(self, table):
        path = self._dataset_path + "/raw-tables/" + table 
        pd_data = self._load_table_from_parquet(path)

        print(f"Table {table} was loaded from cache")
        return pd_data
    
    def save_table_for_analysis(self, table, pd_data):
        path = self._dataset_path + "/analysis-tables/" + table
        self._save_table_to_parquet(path, pd_data)
        
        print(f"Table {table} was saved into parquet cache")
        
    def load_table_for_analysis(self, table):
        path = self._dataset_path + "/analysis-tables/" + table 
        pd_data = self._load_table_from_parquet(path)

        print(f"Table {table} was loaded from cache")
        return pd_data
    
    def _save_raw_table(self, table, pd_data):
        path = self._dataset_path + "/raw-tables/" + table 
        self._save_table_to_parquet(path, pd_data)
        
        print(f"Table {table} was saved into parquet cache")

    def _load_table_from_parquet(self, path):
        pd_data = pd.read_parquet(path)
        return pd_data

    def _save_table_to_parquet(self, path, pd_data):
            pd_data.to_parquet(path)

    def _strip_accents(self, text):
        return ''.join(c for c in unicodedata.normalize('NFKD', text) if unicodedata.category(c) != 'Mn')
    
    def _load_json(self, path):
        with open(path, 'r', encoding="utf-8") as json_file:
            print(f"I have succesfully loaded {path}")
            loaded_json = json.load(json_file)
            return loaded_json

    def _load_yaml(self, path):
        with open(path, "r", encoding='utf-8') as yaml_file:
            print(f"I have succesfully loaded {path}")
            yaml_content = yaml.safe_load(yaml_file)
            return yaml_content

    def _load_excel(self, path, sheet_name, skiprows):
        pd_data = pd.read_excel(path, sheet_name=sheet_name, header=1, skiprows=skiprows)
        print(f"I have succesfully loaded {path}")
        return pd_data

    
        
        