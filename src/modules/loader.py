import pandas as pd
import json
import unicodedata
from connections.database_conn import AzureLoader


class DatasetLoader():
    def __init__(self,
                azure_connection_string,
                list_of_tables_path,
                dataset_path):

        self._azure_loader = AzureLoader(azure_connection_string)
        self._dataset_path = dataset_path

        with open(list_of_tables_path, 'r', encoding="utf-8") as json_config:
            self._json_config = json.load(json_config)

    
    def load_and_save_dataset(self):

        for table in self._json_config:

            query = f"SELECT * FROM [L1].[{table}]"
            pd_data = self._azure_loader.fetch_from_database(query)

            pd_data.columns = [self._strip_accents(column) for column in pd_data.columns]


            path = self._dataset_path + "-raw-tables" + "/" + table

            pd_data.to_parquet(path)

            print(f"Table {table} was saved into parquet cache")

    
    def load_and_print_tables_at_raw_data(self):
         for table in self._json_config:
             pd_data = self.load_raw_table_from_dataset(table)
             print(pd_data)

            
    def load_raw_table_from_dataset(self, table):

        if table in self._json_config:
            path = self._dataset_path + "-raw-tables" + "/" + table
            pd_data = pd.read_parquet(path)

            return pd_data

        else:
            ValueError

    def _strip_accents(self, text):
        return ''.join(c for c in unicodedata.normalize('NFKD', text) if unicodedata.category(c) != 'Mn')


    
        
        