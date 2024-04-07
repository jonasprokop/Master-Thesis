import sqlalchemy as sql
import pandas as pd
import json
import re
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
        self._create_classes_dataset()
        self._create_registrations_dataset()
        self._create_final_datasets()

    def _create_final_datasets(self):
        config = self._loader._final_tables
        for table in config:
            table_config = config[table]
            join_table =  self._load_variable_from_(table_config, "table_name") 
            keys = self._load_variable_from_(table_config, "keys") 

            columns_select = self._load_variable_from_(table_config, "columns_select") 
            columns_rename = self._load_variable_from_(table_config, "columns_rename") 

            pd_data_table = self._loader.load_table_for_analysis(table)
            pd_data_join_table = self._loader.load_table_for_analysis(join_table)

            self._populate_table_with_data(table, pd_data_table)
            self._populate_table_with_data(join_table, pd_data_join_table)

            join_statement = self._create_join_statement([table, join_table], keys)

            pd_data = self._execute_statement(join_statement)

            pd_data = self._deserialize_json_columns(pd_data)
            
            if columns_select:
                pd_data = pd_data[columns_select]

            if columns_rename:
                pd_data = pd_data.rename(columns=columns_rename)

            table += "_joined"

            self._loader.save_table_for_analysis(table, pd_data)

            print(f"{table} was created and saved into parquet")


    def _create_subjects_dataset(self):
        config = self._loader._subjects_dataset_tables
        operations =  self._loader._subjects_dataset_operations
        table_name =  self._load_variable_from_(operations, "table_name") 
        key = self._load_variable_from_(operations, "key") 
        where_statement = self._load_variable_from_(operations, "where_statement") 
        columns_select = self._load_variable_from_(operations, "columns_select") 
        columns_rename = self._load_variable_from_(operations, "columns_rename") 


        for table, addit_info in config.items():

            pd_data = self._loader.load_raw_table(table)

            pd_data = self._preprocess_data(pd_data, table, addit_info)

            self._populate_table_with_data(table, pd_data)

            join_statement = self._create_join_statement(config, key, where_statement)

            pd_data = self._execute_statement(join_statement)

            pd_data = self._deserialize_json_columns(pd_data)
            
        if columns_select:
            pd_data = pd_data[columns_select]

        if columns_rename:
            pd_data = pd_data.rename(columns=columns_rename)


        self._loader.save_table_for_analysis(table_name, pd_data)

        print(f"{table_name} was created and saved into parquet")

    def _create_classes_dataset(self):

        config = self._loader._classes_dataset_tables
        operations =  self._loader._classes_dataset_operations
        table_name =  self._load_variable_from_(operations, "table_name") 
        key = self._load_variable_from_(operations, "key") 
        columns_select = self._load_variable_from_(operations, "columns_select") 
        columns_rename = self._load_variable_from_(operations, "columns_rename") 

        for table, addit_info in config.items():

            pd_data = self._loader.load_raw_table(table)
            pd_data = self._preprocess_data(pd_data, table, addit_info)

        if columns_select:
            pd_data = pd_data[columns_select]

        if columns_rename:
            pd_data = pd_data.rename(columns=columns_rename)

        self._loader.save_table_for_analysis(table_name, pd_data)

        print(f"{table_name} was created and saved into parquet")

    def _create_registrations_dataset(self):

        config = self._loader._registration_dataset_tables
        operations =  self._loader._registration_dataset_operations
        table_name =  self._load_variable_from_(operations, "table_name") 
        key = self._load_variable_from_(operations, "key") 
        columns_select = self._load_variable_from_(operations, "columns_select") 
        columns_rename = self._load_variable_from_(operations, "columns_rename")  

        
        for table, addit_info in config.items():

            pd_data = self._loader.load_raw_table(table)
            
            pd_data = self._preprocess_data(pd_data, table, addit_info)

        if columns_select:
            pd_data = pd_data[columns_select]

        if columns_rename:
            pd_data = pd_data.rename(columns=columns_rename)


        self._loader.save_table_for_analysis(table_name, pd_data)

        print(f"{table_name} was created and saved into parquet")


    def _preprocess_data(self, pd_data, table, addit_info):
        pivot = self._load_variable_from_(addit_info, "pivot") 
        remap = self._load_variable_from_(addit_info, "remap") 
        agg = self._load_variable_from_(addit_info, "agg") 
        split = self._load_variable_from_(addit_info, "split") 
        bool_map_pivot = self._load_variable_from_(addit_info, "bool_map_pivot") 
        average = self._load_variable_from_(addit_info, "average") 
        avg = self._load_variable_from_(addit_info, "avg") 

        recode_categorical_variables = self._load_variable_from_(addit_info, "recode_categorical_variables") 
        recode_day_column = self._load_variable_from_(addit_info, "recode_day_column") 
        recode_time_column = self._load_variable_from_(addit_info, "recode_time_column") 
        split_melt = self._load_variable_from_(addit_info, "split_melt") 
        sum_and_subtract = self._load_variable_from_(addit_info, "sum_and_subtract") 
        

        if average:
            column_name_average = self._load_variable_from_(addit_info, "column_name_average") 
            columns_to_average = self._load_variable_from_(addit_info, "columns_to_average") 
            pd_data = self._average_columns(pd_data, column_name_average, columns_to_average)

        if remap:
            # pivot with dimensionality reduction based on join to other table
            remap_columns = self._load_variable_from_(addit_info, "remap_columns")
            remap_tables = self._load_variable_from_(addit_info, "remap_tables")
            remap_keys = self._load_variable_from_(addit_info, "remap_keys")

            for column in remap_columns:
                remap_table = remap_tables[column]
                remap_key = remap_keys[column]
                pd_data_mapper = self._loader.load_raw_table(remap_table)
                pd_data = self._remap_column(pd_data, pd_data_mapper, column, remap_key)


        if split:
            # splits column into 2 colums
            split_column = self._load_variable_from_(addit_info, "split_column")
            new_column_1 = self._load_variable_from_(addit_info, "new_column_1")
            new_column_2 = self._load_variable_from_(addit_info, "new_column_2")
            split_pattern = self._load_variable_from_(addit_info, "split_pattern")
            pd_data = self._split_column(pd_data, split_column, new_column_1, new_column_2, split_pattern)

        if split_melt:
            exclude_cols = self._load_variable_from_(addit_info, "exclude_cols")
            first_value_cols = self._load_variable_from_(addit_info, "first_value_cols")
            pd_data = self._split_melt(pd_data, exclude_cols, first_value_cols)

        if agg:
            # aggregation function multiple rows aggregate data into one row single column
            id = self._load_variable_from_(addit_info, "agg_index")
            agg_columns = self._load_variable_from_(addit_info, "agg_columns")
            pd_data = self._agg_pivot(pd_data, id, agg_columns)


        if avg:
            id = self._load_variable_from_(addit_info, "avg_index")
            avg_columns = self._load_variable_from_(addit_info, "avg_columns")
            pd_data = self._avg_pivot(pd_data, id, avg_columns)


        if pivot:
            # sum pivot function
            pivot_index = self._load_variable_from_(addit_info, "pivot_index")
            pivot_columns = self._load_variable_from_(addit_info, "pivot_columns")
            pivot_values = self._load_variable_from_(addit_info, "pivot_values")
            aggfunc='sum'
            pd_data = self._pivot_table(pd_data, pivot_index, pivot_columns, pivot_values, aggfunc)

        if bool_map_pivot:
            # bool pivot function
            bool_map_pivot_index = self._load_variable_from_(addit_info, "bool_map_pivot_index")
            bool_map_pivot_columns = self._load_variable_from_(addit_info, "bool_map_pivot_columns")
            pd_data = self._bool_map_pivot_table(pd_data, bool_map_pivot_index, bool_map_pivot_columns)
        
        if sum_and_subtract:
            summation_column = self._load_variable_from_(addit_info, "summation_column")
            columns_to_sum = self._load_variable_from_(addit_info, "columns_to_sum")
            subtracted_column =self._load_variable_from_(addit_info, "subtracted_column")
            columns_to_subtract = self._load_variable_from_(addit_info, "columns_to_subtract")

            pd_data = self._sum_columns(pd_data, summation_column, columns_to_sum)
            pd_data = self._subtract_column(pd_data, subtracted_column, columns_to_subtract)


        if recode_day_column:
            day_column = self._load_variable_from_(addit_info, "day_column")
            pd_data = self._recode_day_column(pd_data, day_column)

        if recode_time_column:
            time_column_start = self._load_variable_from_(addit_info, "time_column_start")
            time_column_end =self._load_variable_from_(addit_info, "time_column_end")
            pd_data = self._recode_time_columns(pd_data, time_column_start, time_column_end)

        if recode_categorical_variables:
            categorical_columns = self._load_variable_from_(addit_info, "categorical_columns")
            recoding_dicts = self._load_variable_from_(addit_info, "recoding_dicts")
            pd_data = self._recode_categorical_columns(pd_data, categorical_columns, recoding_dicts)

        return pd_data
    
    def _load_variable_from_(self, dictionary, variable):
        try:
            value = dictionary[variable]
            return value
        except:
            return None

    
    def _sum_columns(self, pd_data, summation_column, columns_to_sum):
        pd_data[summation_column] = pd_data[columns_to_sum].sum(axis=1)
        return pd_data

    def _subtract_column(self, pd_data, subtracted_column, columns_to_subtract):
        pd_data[subtracted_column] = pd_data[columns_to_subtract[0]] - pd_data[columns_to_subtract[1]]
        return pd_data

    
    def _agg_pivot(self, pd_data, agg_index, values):

        values_list = {value: 'sum' for value in values}
        pd_data = pd_data.groupby(agg_index).agg(values_list).reset_index()

        return pd_data
    
    def _avg_pivot(self, pd_data, avg_index, values):

        values_list = {value: 'mean' for value in values}
        pd_data = pd_data.groupby(avg_index).agg(values_list).reset_index()

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
    
    
    def _split_column(self, pd_data, split_column, new_column_1, new_column_2, split_pattern):
        split_values = pd_data[split_column].str.split(split_pattern, expand=True)
        pd_data[new_column_1] = split_values[0]  
        try:
            pd_data[new_column_2] = split_values[1] 
        except:
            print(split_values[0])
            
        return pd_data
    
    def _average_columns(self, pd_data, column_name, columns_to_average):

        pd_data[column_name] = pd_data[columns_to_average].mean(axis=1)

        return pd_data
    
    
    def _recode_categorical_columns(self, pd_data, columns, recoding_dicts):
        for column in columns:
            recoding_dict = recoding_dicts[column]
            pd_data[column] = pd_data[column].map(recoding_dict)
        return pd_data
    
    def _encode_day(self, day_str):
        if day_str:
            days = ["Po", "Út", "St", "Čt", "Pá", "So", "Ne"]
            date_pattern = r'\d{1,2}\.\d{1,2}' 
            for index, day in enumerate(days):
                contains_date = bool(re.search(date_pattern, day_str))
                if day in day_str:
                    return index + 1, contains_date
            return None, contains_date
        else: 
            return None, False

    def _recode_day_column(self, pd_data, day_column):
        pd_data[['encoded_day', 'singular event']] = pd_data[day_column].apply(lambda x: pd.Series(self._encode_day(x)))
        return pd_data

    def _recode_time_columns(self, pd_data, time_column_start, time_column_end):

        pd_data = self._recode_time_column(pd_data, time_column_start)
        pd_data = self._recode_time_column(pd_data, time_column_end)

        return pd_data

    def _recode_time_column(self, pd_data, column):

        new_column_name = column + "_int"
        pd_data[new_column_name] = pd_data[column].apply(self._recode_time_string)
        return pd_data

        
    def _recode_time_string(self, time_string):
        if time_string:
            hours, minutes = map(int, time_string.split(':'))
        
            total_seconds = hours * 3600 + minutes * 60
            return total_seconds
        else:
            return None
        
    def _split_melt(self, pd_data, exclude_cols, first_value_cols):
            
        new_data = {col: [] for col in pd_data.columns} 
        for index, row in pd_data.iterrows():
            values_to_split = {} 
            for col in pd_data.columns:
                if col in exclude_cols:
                    continue 
                values = str(row[col]).split("\n") 
                if len(values) > 1:
                    values_to_split[col] = values  
            
            if not values_to_split:  
                for col in pd_data.columns:
                    new_data[col].append(row[col])
                continue
            
            max_length = max(len(vals) for vals in values_to_split.values())
            for i in range(max_length):
                for col, values in values_to_split.items():
                    if col in first_value_cols:
                        new_data[col].append(values[0])  
                    else:
                        new_data[col].append(values[i] if i < len(values) else '') 
                for col in pd_data.columns:
                    if col not in values_to_split:
                        new_data[col].append(row[col])
        pd_data = pd.DataFrame(new_data)
        return pd_data

    def _create_join_statement(self, config, keys, where_statement=None):
        subquery = 1
        first_table  = ""         
        join_statement = f"SELECT * "
        first_join = ""
        second_join = ""
        for index, table in enumerate(config):
            if len(keys) == 1:
                key = keys[0]
                first_join = f"""ON {first_table}."{key}" = {table}."{key}" """
                second_join = f"ON subquery_1.{first_table}_{key} = subquery_{subquery}.{table}_{key} "
            if len(keys) == 2:
                key_1 = keys[0]
                key_2 = keys[1]
                first_join = f"""
                                ON {first_table}."{key_1}" = {table}."{key_1}" AND
                                {first_table}."{key_2}" = {table}."{key_2}" 
                """
                second_join = f"""
                                ON subquery_1."{first_table}_{key_1}" = subquery_{subquery}."{table}_{key_1}" AND
                                subquery_1."{first_table}_{key_2}" = subquery_{subquery}."{table}_{key_2}"
                """
            if len(keys) == 3:
                key_1 = keys[0]
                key_2 = keys[1]
                key_3 = keys[3]
                first_join = f"""
                                ON {first_table}."{key_1}" = {table}."{key_1}" AND
                                {first_table}."{key_2}" = {table}."{key_2}" AND
                                {first_table}."{key_3}" = {table}."{key_3}" 
                """
                second_join = f"""
                                ON subquery_1."{first_table}_{key_1}" = subquery_{subquery}."{table}_{key_1}" AND
                                subquery_1."{first_table}_{key_2}" = subquery_{subquery}."{table}_{key_2}" AND
                                subquery_1."{first_table}_{key_3}" = subquery_{subquery}."{table}_{key_3}"
                """
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
                join_statement += first_join
                if where_statement:
                    join_statement += where_statement.format(first_table)
                join_statement += f") AS subquery_1 "
                subquery += 1

            else:
                join_statement += f"LEFT JOIN ( SELECT "
                join_statement = self._create_table_aliases(table, columns, join_statement, index)
                join_statement += f"FROM {table} as {table} "
                join_statement += f") AS subquery_{subquery} "
                join_statement += second_join
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

    def _execute_statement(self, statement):
        query = sql.text(statement)
        with self._sqlalchemy_engine.begin() as session:
            result = session.execute(query)  
            pd_data = pd.DataFrame(result.fetchall(), columns=result.keys())
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






    

        
