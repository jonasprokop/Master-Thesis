import pandas as pd
import json
import sqlalchemy as sql
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import unicodedata


from loaders.database_conn import AzureLoader


class DatasetLoader():
    def __init__(self,
                azure_connection_string,
                list_of_tables_path,
                dataset_path):

        self._azure_loader = AzureLoader(azure_connection_string)
        self._dataset_path = dataset_path

        with open(list_of_tables_path, 'r', encoding="utf-8") as json_config:
            self._json_config = json.load(json_config)

        self._sqlalchemy_engine = sql.create_engine("sqlite:///:memory:")
        self._session = sessionmaker(bind=self._sqlalchemy_engine)
        self._base = declarative_base()
        self._models = {}

        self.create_model_metadata()
        self.populate_model_with_data()
        

    def load_and_save_dataset(self):

        for table in self._json_config:

            query = f"SELECT * FROM {table}"
            pd_data = self._azure_loader.fetch_from_database(query)

            path = self._dataset_path + "-raw-tables" + "/" + table

            pd_data.to_parquet(path)

            print(f"Table {table} was saved into parquet cache")
            
    def load_raw_table_from_dataset(self, table):

        if table in self._json_config:
            path = self._dataset_path + "-raw-tables" + "/" + table
            pd_data = pd.read_parquet(path)

            return pd_data

        else:
            ValueError


    def load_and_print_tables_at_raw_data(self):
         for table in self._json_config:
             pd_data = self.load_raw_table_from_dataset(table)
             print(pd_data)

    def create_model_metadata(self):
        for table_name, columns in self._json_config.items():
            class_name = table_name.replace('.', '_')
            attrs = {'__tablename__': class_name, 'generated_unique_row_id': Column(Integer, primary_key=True)} 
            for column_name, addit_info in columns.items():
                column_name = self._strip_accents(column_name)
                if column_name != 'generated_unique_row_id':
                    attrs[column_name] = Column(String) 
            model_class = type(class_name, (self._base,), attrs)
            self._models[table_name] = model_class
        self._base.metadata.create_all(self._sqlalchemy_engine)

    def populate_model_with_data(self):
        for table in self._json_config:
            pd_data = self.load_raw_table_from_dataset(table)
            model_class = self._models[table]
            for index, row in pd_data.iterrows():
                modified_row = {}
                for column_name, value in row.items():
                    modified_column_name = self._strip_accents(column_name)
                    modified_row[modified_column_name] = value
                model_instance = model_class(**modified_row)
            self._session.add(model_instance)
            self._session.commit()
          
    def create_model(self):
        return self._sqlalchemy_engine, self._session
    










    def _strip_accents(self, text):
        return ''.join(c for c in unicodedata.normalize('NFKD', text) if unicodedata.category(c) != 'Mn')
    
        
        