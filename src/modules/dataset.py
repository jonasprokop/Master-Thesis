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
        self._base = declarative_base()
        self._models = {}
        self._create_model_metadata()  
        self._session_factory = sessionmaker(bind=self._sqlalchemy_engine)
        self._session = self._session_factory()
        self._base.metadata.bind = self._sqlalchemy_engine 
        self._populate_model_with_data()


    def _create_model_metadata(self):
        for table_name, columns in self._dataset_loader._json_config.items():
            class_name = table_name.replace('.', '_')
            attrs = {'__tablename__': class_name} 
            for column_name, addit_info in columns.items():

                column_name = self._dataset_loader._strip_accents(column_name)
                data_type = addit_info['data_type']
                primary_key = addit_info['primary_key']
                has_primary_key = False

                if data_type and primary_key:
                    column_type = getattr(sql, data_type)
                    attrs[column_name] = Column(column_type, primary_key=True)
                    has_primary_key = True

                elif data_type and not primary_key:
                    column_type = getattr(sql, data_type)
                    attrs[column_name] = Column(column_type)

                elif not data_type and primary_key:
                    attrs[column_name] = Column(String, primary_key=True) 
                    has_primary_key = True
                
                elif not data_type and not primary_key:
                    attrs[column_name] = Column(String) 

            if not has_primary_key:
                attrs["generated_unique_row_id"] = Column(Integer, primary_key=True)

            model_class = type(class_name, (self._base,), attrs)
            self._models[table_name] = model_class
        self._base.metadata.create_all(self._sqlalchemy_engine)

    def _populate_model_with_data(self):
        for table in self._dataset_loader._json_config:
            if table not in ["bilance_rozsireno", "predmety_total_Hkat_rozsireno"]:
                pd_data = self._dataset_loader.load_raw_table_from_dataset(table)
                model_class = self._models[table]

                for index, row in pd_data.iterrows():
                    try:
                        if hasattr(model_class, 'generated_unique_row_id'):
                            model_instance.generated_unique_row_id = index              
                        model_instance = model_class(**row)
                        self._session.add(model_instance)
                        self._session.commit()
                    except:
                        print("Nonunique row found")


                print(f"Model was populated with table {table}")
          
    
    def create_test_dataset(self):
        for query_name, query_string in self._dataset_loader._transformation_data.items():
            query = sql.text(query_string)

            with self._sqlalchemy_engine.begin() as session:
                result = session.execute(query)  
                for row in result:
                    print(row)