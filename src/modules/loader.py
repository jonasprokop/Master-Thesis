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
        final_tables = os.environ.get("FINAL_TABLES")
        statistical_analysis = os.environ.get("STATISTICAL_ANALYSIS")
        additional_data = os.environ.get("ADDITIONAL_DATA")
        model_data = os.environ.get("MODEL_DATA")
        
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
        self._final_tables = self._load_yaml(final_tables)
        self._statistical_analysis = self._load_json(statistical_analysis)
        self._additional_data = self._load_json(additional_data)
        self._model_data = self._load_yaml(model_data)

    def load_and_save_dataset(self):
        self._save_tables_from_db(self._mapper)
        self._save_tables_from_excel(self._excel_data)

    
    def _save_tables_from_db(self, config): 
        mapper = config["mapper"]
        where_statement_config = config["where_statement_config"]
        where_statement_tables = where_statement_config["where_statement_tables"]
        for table, database_table_name  in mapper.items():
            query = f"""SELECT * FROM {database_table_name}"""
            if table in where_statement_tables:
                where_statement_table_config = where_statement_config[table]
                where_statement_column = where_statement_table_config["where_statement_column"]
                where_statement_condition = where_statement_table_config["where_statement_condition"]
                query += f"""WHERE "{where_statement_column}" in {where_statement_condition}"""

            pd_data = self._azure_loader.fetch_from_database(query)

            print(f"Table {table} was downloaded from source db")

            pd_data.columns = [self._strip_accents(column) for column in pd_data.columns]

            self._save_raw_table(table, pd_data)

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
                pd_data = pd.concat(partial_tables, axis=0, ignore_index=True, join='outer')
                pd_data.columns = [self._strip_accents(column) for column in pd_data.columns]
                print(f"I have succesfully concatened {table}")
                self._save_raw_table(table, pd_data)


    def _load_additional_data_for_last_year(self, pd_data, table):

        if table in self._additional_data: 
            config = self._additional_data[table]
            pd_data_additional = self._load_excel(config["path"], sheet_name="Sheet1", skiprows=5)
            pd_data_additional.columns = [self._strip_accents(column) for column in pd_data_additional.columns]

            if config["append"]:
                additional_data = pd_data_additional

            elif config["split"]:
                additional_data = pd_data_additional[config["split_column"]]

            else:
                additional_data = None

        else:
            additional_data =  None

        if additional_data is not None:
            new_pd_data = [pd_data, additional_data]
            pd_data = pd.concat(new_pd_data, axis=0, ignore_index=True, join='outer')

        return pd_data

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
        pd_data = pd.read_excel(path, sheet_name=sheet_name, header=0, skiprows=skiprows)
        print(f"I have succesfully loaded {path}")
        return pd_data

    
        
        