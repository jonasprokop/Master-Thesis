import sqlalchemy as sql
import pandas as pd
import json
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
        #self._create_subjects_dataset()
        self._create_classes_dataset()
        self._create_registrations_dataset()

    def _create_subjects_dataset(self):
        config = self._loader._subjects_dataset_tables
        operations =  self._loader._subjects_dataset_operations
        table_name =  operations["table_name"] 
        key = operations["key"] 
        where_statement = operations["where_statement"] 
        columns_select = operations["columns_select"] 

        for table, addit_info in config.items():

            pd_data = self._loader.load_raw_table(table)

            pd_data = self._preprocess_data(pd_data, table, addit_info)

            self._populate_table_with_data(table, pd_data)

            join_statement = self._create_join_statement(config, key, where_statement)

            pd_data = self._execute_statement(join_statement)

            pd_data = self._deserialize_json_columns(pd_data)
            
        #if columns_select:
            #pd_data = pd_data[[columns_select]]

        self._loader.save_table_for_analysis(table_name, pd_data)

        print(f"{table_name} was created and saved into parquet")

    def _create_classes_dataset(self):

        config = self._loader._classes_dataset_tables
        operations =  self._loader._classes_dataset_operations
        table_name =  operations["table_name"] 
        columns_select = operations["columns_select"] 

        for table, addit_info in config.items():

            pd_data = self._loader.load_raw_table(table)
            pd_data = self._preprocess_data(pd_data, table, addit_info)

            #if columns_select:
            #   pd_data = pd_data[[columns_select]]

            self._loader.save_table_for_analysis(table_name, pd_data)

        print(f"{table_name} was created and saved into parquet")

    def _create_registrations_dataset(self):

        config = self._loader._registration_dataset_tables
        operations =  self._loader._registration_dataset_operations
        table_name =  operations["table_name"] 
        columns_select = operations["columns_select"] 

        
        for table, addit_info in config.items():

            pd_data = self._loader.load_raw_table(table)


            pd_data = self._preprocess_data(pd_data, table, addit_info)

            #if columns_select:
            #   pd_data = pd_data[[columns_select]]

            self._loader.save_table_for_analysis(table_name, pd_data)

        print(f"{table_name} was created and saved into parquet")

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


    def _populate_table_with_data(self, table, pd_data):
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

    def _preprocess_data(self, pd_data, table, addit_info):
        pivot = addit_info["pivot"]
        remap = addit_info["remap"]
        agg = addit_info["agg"]
        split = addit_info["split"]
        bool_map_pivot = addit_info["bool_map_pivot"]
        average = addit_info["average"]


        if average:
            column_name_average = addit_info["column_name_average"]
            columns_to_average = addit_info["columns_to_average"]
            pd_data = self._sum_columns_into_average(self, pd_data, column_name_average, columns_to_average)

        if remap:
            # pivot with dimensionality reduction based on join to other table
            remap_columns = addit_info["remap_columns"]
            remap_tables = addit_info["remap_tables"]
            remap_keys = addit_info["remap_keys"]

            for column in remap_columns:
                remap_table = remap_tables[column]
                remap_key = remap_keys[column]
                pd_data_mapper = self._loader.load_raw_table(remap_table)
                pd_data = self._remap_column(pd_data, pd_data_mapper, column, remap_key)


        if split:
            # splits column into 2 colums
            split_column = addit_info["split_column"]
            split_patterns = addit_info["split_patterns"]
            pd_data = self._split_column(pd_data, split_column, split_patterns)

        if agg:
            # aggregation function multiple rows aggregate data into one row single column
            id = addit_info["agg_index"]
            agg_values = addit_info["agg_columns"]
            pd_data = self._agg_pivot(pd_data, id, agg_values)

        if pivot:
            # sum pivot function
            index = addit_info["pivot_index"]
            columns = addit_info["pivot_columns"]
            values = addit_info["pivot_values"]
            aggfunc='sum'
            pd_data = self._pivot_table(pd_data, index, columns, values, aggfunc)

        if bool_map_pivot:
            # bool pivot function
            index = addit_info["bool_map_pivot_index"]
            columns = addit_info["bool_map_pivot_columns"]
            pd_data = self._bool_map_pivot_table(pd_data, index, columns)

        return pd_data
        

    def _agg_pivot(self, pd_data, id, values):

        values_list = {value: 'sum' for value in values}
        pd_data = pd_data.groupby(id).agg(values_list).reset_index()

        return pd_data
    


    def _pivot_table(self, pd_data, index, columns, values, aggfunc):

        pd_data = pd_data.pivot_table(index=index, columns=columns, values=values, aggfunc=aggfunc, fill_value=0)
        pd_data = pd_data.reset_index()
        
        pd_data.columns = [f'{col[1]}_{col[0]}' if isinstance(col, tuple) else col for col in pd_data.columns]
        pd_data.columns = [col.strip('_') for col in pd_data.columns]
        pd_data.columns = [self._loader._strip_accents(column) for column in pd_data.columns]
        
        return pd_data
    
    def _bool_map_pivot_table(self, pd_data, index, columns):
        agg_func = 'first'

        one_hot = pd.get_dummies(pd_data[columns], prefix=columns) 
        pd_data = pd.concat([pd_data, one_hot], axis=1)

        pd_data = pd_data.groupby(index).agg(agg_func)
        pd_data.drop(columns=[columns], inplace=True)

        pd_data.reset_index(inplace=True)

        return pd_data
        
    def _remap_column(self, pd_data, pd_data_mapper, remap_column, remap_keys):

        pd_data[remap_column] = pd_data[remap_column].map(pd_data_mapper.set_index(remap_keys[0])[remap_keys[1]])

        return pd_data
    
    
    def _split_column(self, pd_data, split_column, split_patterns):
        for pattern, column in split_patterns.items():
            split_values = pd_data[split_column].str.split(pattern, expand=True)
            pd_data[column] = split_values[0]  
            try:
                pd_data[split_column] = split_values[1] 
            except:
                print(split_values[0])
            
        return pd_data
    
    def _sum_columns_into_average(self, pd_data, column_name, columns_to_average):

        pd_data[column_name] = pd_data[columns_to_average].mean(axis=1)

        return pd_data
    
    def _encode_day(self, day_str):
        days = ["Po", "Út", "St", "Čt", "Pá", "So", "Ne"]
        try:
            return days.index(day_str.split()[0]) + 1
        except:
            return None
    
    def _enconde_daytime_interval(self, start_time, end_time):
        intervals = {
        1: ("06:00", "9:00"),
        2: ("9:00", "13:00"),
        3: ("13:00", "16:00"),
        4: ("16:00", "19:00"),
        5: ("19:00", "23:59")
        }
        #tady by to chtělo projet normální rozdělení intervalů
        for interval, (start, end) in intervals.items():
            try:
                if pd.to_datetime(start_time).time() <= pd.to_datetime(start).time() and pd.to_datetime(end_time).time() <= pd.to_datetime(end).time():
                    return interval
            except:
                return None
            
    def _generate_combination_dict(self):
        combination_dict = {}
        for day in range(1, 8):
            for interval in range(1, 6):
                combination_dict[(day, interval)] = (day - 1) * 5 + interval
        return combination_dict
    
    def _assign_schedulge_key(self, combination_dict, day, interval):
        try:
            key = combination_dict[(day, interval)]
            return key
        except:
            return None
        
    def _recode_subject_time_intervals(self, pd_data, day_column, start_column, end_column):

        day_column_encoded = day_column + "_encoded"
        interval_column = "interval_column"
        schedulge_code = "schedulge_code"

        pd_data[day_column_encoded] = pd_data[day_column].apply(self._encode_day)
        pd_data[interval_column] = pd_data.apply(lambda x: self._enconde_daytime_interval(x[start_column], x[end_column]), axis=1)
        combination_dict = self._generate_combination_dict()
        pd_data[schedulge_code] = pd_data.apply(lambda x: self._assign_schedulge_key(combination_dict, x[day_column_encoded], x[interval_column]), axis=1)

        pd_data.drop([start_column, end_column, interval_column, day_column_encoded])

        return pd_data

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
                join_statement += f"JOIN {table} as {table} "
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
    
    def _deserialize_json_columns(self, pd_data):
        for column in pd_data.columns:
            pd_data[column] = pd_data[column].apply(self._parse_json_string)
        return pd_data
    
    def _parse_json_string(self, string):
        try:
            return json.loads(string)
        except (json.JSONDecodeError, TypeError):
            return string  






    

        
