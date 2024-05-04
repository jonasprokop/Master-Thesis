import sqlalchemy as sql
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd

from modules.operations import DatasetOperations

class DatasetMaker(DatasetOperations):
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


    def _create_subjects_dataset(self):
        config = self._loader._subjects_dataset_tables
        operations =  self._loader._subjects_dataset_operations
        table_name =  self._load_variable_from(operations, "table_name") 
        key = self._load_variable_from(operations, "key") 
        where_statement = self._load_variable_from(operations, "where_statement") 

        for table, addit_info in config.items():

            pd_data = self._loader.load_raw_table(table)

            pd_data = self._add_code_columns(addit_info, pd_data)

            pd_data = self._add_new_categorical_columns(addit_info, pd_data)

            pd_data = self._loader._load_additional_data_for_last_year(pd_data, table)

            pd_data = self._preprocess_data(pd_data, table, addit_info)

            self._populate_table_with_data(table, pd_data)

            join_statement = self._create_join_statement(config, key, where_statement)

        pd_data = self._execute_statement(join_statement)

        pd_data = self._deserialize_json_columns(pd_data)

        pd_data = self._post_process_data(table, operations, pd_data)

        self._loader.save_table_for_analysis(table_name, pd_data)

        print(f"{table_name} was created and saved into parquet")

    def _create_classes_dataset(self):
        config = self._loader._classes_dataset_tables
        operations =  self._loader._classes_dataset_operations
        table_name =  self._load_variable_from(operations, "table_name") 

        for table, addit_info in config.items():

            pd_data = self._loader.load_raw_table(table)
            pd_data = self._preprocess_data(pd_data, table, addit_info)

        pd_data = self._post_process_data(table, operations, pd_data)

        self._loader.save_table_for_analysis(table_name, pd_data)

        print(f"{table_name} was created and saved into parquet")

    def _create_registrations_dataset(self):

        config = self._loader._registration_dataset_tables
        operations =  self._loader._registration_dataset_operations
        table_name =  self._load_variable_from(operations, "table_name") 
        
        for table, addit_info in config.items():

            pd_data = self._loader.load_raw_table(table)
            pd_data = self._preprocess_data(pd_data, table, addit_info)

        pd_data = self._post_process_data(table, operations, pd_data)

        self._loader.save_table_for_analysis(table_name, pd_data)

        print(f"{table_name} was created and saved into parquet")

    def _create_final_datasets(self):
        config = self._loader._final_tables

        for table in config:
            table_config = config[table]
            join_table =  self._load_variable_from(table_config, "table_name") 
            keys = self._load_variable_from(table_config, "keys") 
            left_join =  self._load_variable_from(table_config, "left_join") 

            pd_data_table = self._loader.load_table_for_analysis(table)
            pd_data_join_table = self._loader.load_table_for_analysis(join_table)

            self._populate_table_with_data(table, pd_data_table)

            self._populate_table_with_data(join_table, pd_data_join_table)

            join_statement = self._create_join_statement([table, join_table], keys, left_join=left_join)

            pd_data = self._execute_statement(join_statement)

            pd_data = self._deserialize_json_columns(pd_data)

            pd_data = self._post_process_data(table, table_config, pd_data)

            table_name_final = table + "_final"

            self._loader.save_table_for_analysis(table_name_final, pd_data)

            print(f"{table} was created and saved into parquet")


    def _preprocess_data(self, pd_data, table, addit_info):
        pivot = self._load_variable_from(addit_info, "pivot") 
        remap = self._load_variable_from(addit_info, "remap") 
        agg = self._load_variable_from(addit_info, "agg") 
        split = self._load_variable_from(addit_info, "split") 
        bool_map_pivot = self._load_variable_from(addit_info, "bool_map_pivot") 
        average = self._load_variable_from(addit_info, "average") 
        avg = self._load_variable_from(addit_info, "avg") 
        split_melt = self._load_variable_from(addit_info, "split_melt") 
        sum_and_subtract = self._load_variable_from(addit_info, "sum_and_subtract") 

        if average:
            column_name_average = self._load_variable_from(addit_info, "column_name_average") 
            columns_to_average = self._load_variable_from(addit_info, "columns_to_average") 
            pd_data = self._average_columns(pd_data, column_name_average, columns_to_average)

        if remap:
            remap_columns = self._load_variable_from(addit_info, "remap_columns")
            remap_tables = self._load_variable_from(addit_info, "remap_tables")
            remap_keys = self._load_variable_from(addit_info, "remap_keys")

            for column in remap_columns:
                remap_table = remap_tables[column]
                remap_key = remap_keys[column]
                pd_data_mapper = self._loader.load_raw_table(remap_table)
                pd_data = self._remap_column(pd_data, pd_data_mapper, column, remap_key)


        if split:
            split_column = self._load_variable_from(addit_info, "split_column")
            new_column_1 = self._load_variable_from(addit_info, "new_column_1")
            new_column_2 = self._load_variable_from(addit_info, "new_column_2")
            split_pattern = self._load_variable_from(addit_info, "split_pattern")
            pd_data = self._split_column(pd_data, split_column, new_column_1, new_column_2, split_pattern)

        if split_melt:
            exclude_cols = self._load_variable_from(addit_info, "exclude_cols")
            first_value_cols = self._load_variable_from(addit_info, "first_value_cols")
            pd_data = self._split_melt(pd_data, exclude_cols, first_value_cols)

        if agg:
            id = self._load_variable_from(addit_info, "agg_index")
            agg_columns = self._load_variable_from(addit_info, "agg_columns")
            pd_data = self._aggregate(pd_data, id, agg_columns)

        if avg:
            id = self._load_variable_from(addit_info, "avg_index")
            avg_columns = self._load_variable_from(addit_info, "avg_columns")
            pd_data = self._average_in_target_value(pd_data, id, avg_columns)

        if pivot:
            pivot_index = self._load_variable_from(addit_info, "pivot_index")
            pivot_columns = self._load_variable_from(addit_info, "pivot_columns")
            pivot_values = self._load_variable_from(addit_info, "pivot_values")
            aggfunc='sum'
            pd_data = self._pivot_table(pd_data, pivot_index, pivot_columns, pivot_values, aggfunc)

        if bool_map_pivot:
            bool_map_pivot_index = self._load_variable_from(addit_info, "bool_map_pivot_index")
            bool_map_pivot_columns = self._load_variable_from(addit_info, "bool_map_pivot_columns")
            pd_data = self._bool_map_pivot_table(pd_data, bool_map_pivot_index, bool_map_pivot_columns)
        
        if sum_and_subtract:
            summation_column = self._load_variable_from(addit_info, "summation_column")
            columns_to_sum = self._load_variable_from(addit_info, "columns_to_sum")
            subtracted_column =self._load_variable_from(addit_info, "subtracted_column")
            columns_to_subtract = self._load_variable_from(addit_info, "columns_to_subtract")

            pd_data = self._sum_columns(pd_data, summation_column, columns_to_sum)
            pd_data = self._subtract_columns(pd_data, subtracted_column, columns_to_subtract)

        return pd_data
    

    def _post_process_data(self, table, table_config, pd_data):
        columns_select = self._load_variable_from(table_config, "columns_select") 
        columns_rename = self._load_variable_from(table_config, "columns_rename") 

        save_not_recoded = self._load_variable_from(table_config, "save_not_recoded") 
        recode_categorical_variables = self._load_variable_from(table_config, "recode_categorical_variables") 
        recode_day_column = self._load_variable_from(table_config, "recode_day_column") 
        recode_time_column = self._load_variable_from(table_config, "recode_time_column")
        fill_nan_values =  self._load_variable_from(table_config, "fill_nan_values")
        pca_dim_reduction = self._load_variable_from(table_config, "pca_dim_reduction")
        rescale_data = self._load_variable_from(table_config, "rescale_data")
        

        if columns_select:
            pd_data = pd_data[columns_select]

        if columns_rename:
            pd_data = pd_data.rename(columns=columns_rename)

        if save_not_recoded:
            table_name_not_recoded = table + "_not_recoded"
            self._loader.save_table_for_analysis(table_name_not_recoded, pd_data)
        
        if recode_categorical_variables:
            categorical_columns = self._load_variable_from(table_config, "categorical_columns")
            recoding_dicts = self._load_variable_from(table_config, "recoding_dicts")
            pd_data = self._recode_categorical_columns(pd_data, categorical_columns, recoding_dicts)

        if recode_day_column:
            day_column = self._load_variable_from(table_config, "day_column")
            pd_data = self._recode_day_column(pd_data, day_column)

        if recode_time_column:
            time_column_start = self._load_variable_from(table_config, "time_column_start")
            time_column_end = self._load_variable_from(table_config, "time_column_end")
            pd_data = self._recode_time_columns(pd_data, time_column_start, time_column_end)

        if fill_nan_values:
            pd_data = self._fill_nan_values(pd_data)

        if pca_dim_reduction:
            columns_to_combine = self._load_variable_from(table_config, "columns_to_combine")
            final_dim = self._load_variable_from(table_config, "final_dim")
            new_column_names = self._load_variable_from(table_config, "new_column_names")
            pd_data = self._pca_dim_reduction(pd_data, columns_to_combine, final_dim, new_column_names)

        if rescale_data:
            columns_select_rescale_data = self._load_variable_from(table_config, "columns_select_rescale_data")
            pd_data = self._scale_and_concat(pd_data, columns_select_rescale_data)

        return pd_data



