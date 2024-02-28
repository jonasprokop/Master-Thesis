import sqlalchemy as sql
import pandas as pd
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


class DatasetMaker():
    def __init__(self,
                dataset_loader):
        
        self._dataset_loader = dataset_loader

        self._sqlalchemy_engine = sql.create_engine("sqlite:///:memory:")
        self._session_factory  = sessionmaker(bind=self._sqlalchemy_engine)
        self._session = self._session_factory()
        self._base = declarative_base()
        self._models = {}

    def create_model_metadata(self):
        for table_name, columns in self._dataset_loader._json_config.items():
            class_name = table_name.replace('.', '_')
            attrs = {'__tablename__': class_name, 'generated_unique_row_id': Column(Integer, primary_key=True)} 
            for column_name, addit_info in columns.items():
                column_name = self._dataset_loader._strip_accents(column_name)
                if column_name != 'generated_unique_row_id':
                    attrs[column_name] = Column(String) 
            model_class = type(class_name, (self._base,), attrs)
            self._models[table_name] = model_class
        self._base.metadata.create_all(self._sqlalchemy_engine)

    def populate_model_with_data(self):
        for table in self._dataset_loader._json_config:
            pd_data = self._dataset_loader.load_raw_table_from_dataset(table)
            model_class = self._models[table]
            for index, row in pd_data.iterrows():
                model_instance = model_class(**row)
            self._session.add(model_instance)
            self._session.commit()
            print(f"Model was populated with table {table}")
          
    
