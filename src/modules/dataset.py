import sqlalchemy as sql
import pandas as pd
from sqlalchemy import Column, Integer, String, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


class DatasetMaker():
    def __init__(self,
                loader):
        
        self._loader = loader
        self._sqlalchemy_engine = sql.create_engine("sqlite:///:memory:")
        self._base = declarative_base()
        self._models = {}
        self._create_model_metadata(self._loader._model_metadata)  
        self._session_factory = sessionmaker(bind=self._sqlalchemy_engine)
        self._session = self._session_factory()
        self._base.metadata.bind = self._sqlalchemy_engine 
        self._tables_populated = []

        
    def create_datasets(self):
        self._create_subjects_dataset()

    def _create_subjects_dataset(self):
        self._create_dataset(self._loader._subjects_dataset_tables, self._loader._subjects_dataset_operations)


    def _create_model_metadata(self, config):
        for table_name, columns in config.items():
            class_name = table_name.replace('.', '_')
            attrs = {'__tablename__': class_name} 

            has_primary_key = False

            for column_name, addit_info in columns.items():

                column_name = self._loader._strip_accents(column_name)
                data_type = addit_info['data_type']
                primary_key = addit_info['primary_key']

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


    def _populate_model_with_data(self, table, pd_data):
        if table not in self._tables_populated:
            self._tables_populated.append(table)
            model_class = self._models[table]

            for index, row in pd_data.iterrows():        
                model_instance = model_class(**row)
                if hasattr(model_class, 'generated_unique_row_id'):
                    model_instance.generated_unique_row_id = index      
                self._session.add(model_instance)
                self._session.commit()

            print(f"Model was populated with table {table}")

    def _pivot_table(self, pd_data, index, columns, values, aggfunc):
        df_pivot = pd_data.pivot_table(index=index, columns=columns, values=values, aggfunc=aggfunc)
        
        df_pivot = df_pivot.reset_index()
        
        df_pivot.columns = [f'{col[1]}_{col[0]}' if isinstance(col, tuple) else col for col in df_pivot.columns]
        
        df_pivot.columns = [col[1:] if col.startswith('_') else col for col in df_pivot.columns]
        
        return df_pivot
    def _remap_column(self, pd_data, pd_data_mapper, remap_column, remap_keys):
        pd_data[remap_column] = pd_data[remap_column].map(pd_data_mapper.set_index(remap_keys[0])[remap_keys[1]])


        return pd_data
    
    def _populate_dataset(self, config):
        for table, addit_info in config.items():
            pd_data = self._loader.load_raw_table(table)
            pivot = addit_info["pivot"]
            remap = addit_info["remap"]
            if pivot and remap:
                remap_column = addit_info["remap_column"]
                remap_keys = addit_info["remap_keys"]
                remap_table = addit_info["remap_table"]
                index = addit_info["index"]
                columns = addit_info["columns"]
                values = addit_info["values"]
                aggfunc = 'sum'
                pd_data_mapper = self._loader.load_raw_table(remap_table)
                pd_data = self._remap_column(pd_data, pd_data_mapper, remap_column, remap_keys)
                pd_data = self._pivot_table(pd_data, index, columns, values, aggfunc)

            if pivot and not remap:
                index = addit_info["index"]
                columns = addit_info["columns"]
                values = addit_info["values"]
                aggfunc='first'
                pd_data = self._pivot_table(pd_data, index, columns, aggfunc)
            
            self._populate_model_with_data(table, pd_data)

    def _create_join_statement(self, config, key, where_statement):
        subquery = 1
        first_table  = ""         
        join_statement = f"SELECT * "
        for index, table in enumerate(config):
            table_data = self._loader._model_metadata[table]
            columns = table_data.keys()
            if index == 0:
                join_statement += f"FROM ( SELECT "
                join_statement = self._create_table_aliases(table, columns, join_statement, index)
                first_table = table

            elif index == 1:
                join_statement = self._create_table_aliases(table, columns, join_statement, index)
                join_statement += f"FROM {first_table} as {first_table} "
                join_statement += f"LEFT JOIN {table} as {table} "
                join_statement += f"""ON {first_table}."{key}" = {table}."{key}" """
                if where_statement:
                    join_statement += where_statement.format(first_table)
                join_statement += f") AS subquery_1 "
                subquery += 1

            else:
                join_statement += f"LEFT JOIN ( SELECT "
                join_statement = self._create_table_aliases(table, columns, join_statement, index)
                join_statement += f"FROM {table} as {table} "
                join_statement += f") AS subquery_{subquery} "
                join_statement += f"ON subquery_1.{first_table}_{key} = subquery_{subquery}.{table}_{key} "
                subquery += 1

        return join_statement
    
    def _create_table_aliases(self, table, columns, join_statement, outer_index):
        inner_index = 0
        len_col = len(columns)
        for column in columns:
            inner_index += 1
            column = self._loader._strip_accents(column)
            if inner_index == len_col and (outer_index % 2 != 0 or outer_index != 0):
                join_statement += f"""{table}."{column}" AS "{table}_{column}" """
            else:
                join_statement += f"""{table}."{column}" AS "{table}_{column}", """

        return join_statement


    def _execute_statement(self, statement):
        query = sql.text(statement)
        with self._sqlalchemy_engine.begin() as session:
            result = session.execute(query)  
            pd_data = pd.DataFrame(result.fetchall(), columns=result.keys())
            print(pd_data)
        return pd_data


    def _create_dataset(self, config, operations):
        table_name =  operations["table_name"] 
        key = operations["key"] 
        where_statement = operations["where_statement"] 
        columns_select = operations["columns_select"] 

        self._populate_dataset(config)

        join_statement = self._create_join_statement(config, key, where_statement)

        pd_data = self._execute_statement(join_statement)
        
        #if columns_select:
            #pd_data = pd_data[[columns_select]]

        self._loader.save_table_for_analysis(table_name, pd_data)

        print(f"{table_name} was created and saved into parquet")



    

        
