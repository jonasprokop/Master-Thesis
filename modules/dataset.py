import pandas as pd
import json
import sqlalchemy as sql
from sqlalchemy import Column, Integer, String
from sqlalchemy import declarative_base, sessionmaker



from loaders.database_conn import AzureLoader


class DatasetLoader():
    def __init__(self,
                azure_connection_string,
                dataset_path):

        self._azure_loader = AzureLoader(azure_connection_string)
        self._dataset_path = dataset_path

        with open('tables.json', 'r') as json_config:
            self._json_config = json.load(json_config)

        self._sqlalchemy_engine = sql.create_engine("sqlite:///:memory:")
        self._Session = sessionmaker(bind=self._sqlalchemy_engine)
        self._Base = declarative_base()
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
            attrs = {'__tablename__': class_name}
            for column_name, _ in columns.items():
                attrs[column_name] = Column(String) 
            model_class = type(class_name, (self._Base,), attrs)
            self._models[table_name] = model_class
        self._Base.metadata.create_all(self._sqlalchemy_engine)

    def populate_model_with_data(self):
        for table in self._json_config:
            pd_data = self.load_raw_table_from_dataset(table)
            model_class = self._models[table]
            for index, row in pd_data.iterrows():
                model_instance = model_class(**row)
                self.session.add(model_instance)
            self.session.commit()
          
    def create_model(self):
        return self._sqlalchemy_engine, self._Session
    
        
        