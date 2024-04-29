
import pandas as pd
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler 
import re

from modules.sql_model import DataModel

class DatasetOperations(DataModel):
    def _load_variable_from(self, dictionary, variable):

        try:
            value = dictionary[variable]
            return value
        
        except:
            return None

    def _sum_columns(self, pd_data, summation_column, columns_to_sum):
        pd_data[summation_column] = pd_data[columns_to_sum].sum(axis=1)

        return pd_data

    def _subtract_columns(self, pd_data, subtracted_column, columns_to_subtract):
        pd_data[subtracted_column] = pd_data[columns_to_subtract[0]] - pd_data[columns_to_subtract[1]]
        return pd_data

    def _aggregate(self, pd_data, agg_index, values):

        values_list = {value: 'sum' for value in values}
        pd_data = pd_data.groupby(agg_index).agg(values_list).reset_index()

        return pd_data
    
    def _average_in_target_value(self, pd_data, avg_index, values):

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
        one_hot = pd.get_dummies(pd_data[columns], prefix=columns, dummy_na=False) 
        pd_data = pd.concat([pd_data, one_hot], axis=1)

        pd_data = pd_data.groupby(index).agg(agg_func)
        pd_data.drop(columns=[columns], inplace=True)

        pd_data.reset_index(inplace=True)

        return pd_data
        
    def _remap_column(self, pd_data, pd_data_mapper, remap_column, remap_keys):

        pd_data = self._create_mapped_column(pd_data, pd_data_mapper, remap_column, remap_column, remap_keys)

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
    
    def _recode_day_column(self, pd_data, day_column):
        pd_data[[day_column, 'jednorazova_akce']] = pd_data[day_column].apply(lambda x: pd.Series(self._encode_day(x)))

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

    def _recode_time_columns(self, pd_data, time_column_start, time_column_end):

        pd_data = self._recode_time_column_end(pd_data, time_column_start, time_column_end)
        pd_data = self._recode_time_column_start(pd_data, time_column_start)

        return pd_data

    def _recode_time_column_start(self, pd_data, column):

        pd_data[column] = pd_data[column].apply(self._recode_time_string)

        return pd_data
    
    def _recode_time_column_end(self, pd_data, time_column_start, time_column_end):

        pd_data[time_column_end] = pd_data.apply(lambda row: self._recode_time_interval(row[time_column_start], row[time_column_end]), axis=1)

        return pd_data
    
    def _recode_time_interval(self, start_time, end_time):
        if start_time and end_time:
            start_seconds = self._recode_time_string(start_time)
            end_seconds = self._recode_time_string(end_time)

            duration_seconds = end_seconds - start_seconds
            return duration_seconds
        else:
            return None

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

    def _add_code_columns(self, config, pd_data):
        code_config = self._load_variable_from(config, "new_code_column_code_config") 

        for new_code_column_add_code_colum, part_config in code_config.items():
            old_column_name_add_code_colum = self._load_variable_from(part_config, "old_column_name_add_code_colum") 
            remap_keys_add_code_column = self._load_variable_from(part_config, "remap_keys_add_code_column")
            mapper_add_code_column =  self._load_variable_from(part_config, "mapper_add_code_column")
            pd_data_mapper = self._loader.load_raw_table(mapper_add_code_column)
            pd_data = self._create_mapped_column(pd_data, pd_data_mapper, new_code_column_add_code_colum, old_column_name_add_code_colum, remap_keys_add_code_column)

        return pd_data

    def _add_new_categorical_columns(self, config, pd_data):
        code_config = self._load_variable_from(config, "add_new_categorical_column_code_config") 

        for old_column_name_add_new_categorical_column, part_config in code_config.items():
            if old_column_name_add_new_categorical_column in pd_data.columns:
                new_column_name_add_new_categorical_column = self._load_variable_from(part_config, "new_column_name_add_new_categorical_column") 
                remap_keys_add_new_categorical_column = self._load_variable_from(part_config, "remap_keys_add_new_categorical_column")
                mapper_add_new_categorical_column =  self._load_variable_from(part_config, "mapper_add_new_categorical_column")
                pd_data_mapper = self._loader.load_raw_table(mapper_add_new_categorical_column)
                pd_data = self._create_mapped_column(pd_data, pd_data_mapper, new_column_name_add_new_categorical_column, old_column_name_add_new_categorical_column, remap_keys_add_new_categorical_column)
                            
        return pd_data
    

    def _create_mapped_column(self, pd_data, pd_data_mapper, new_code_column, old_column_name, remap_keys):
        pd_data[new_code_column] = pd_data[old_column_name].map(pd_data_mapper.set_index(remap_keys[0])[remap_keys[1]])
        return pd_data
    
    def _fill_nan_values(self, pd_data):
        pd_data.fillna(value=0, inplace=True)
        return pd_data
    
    def _pca_dim_reduction(self, pd_data, columns_to_combine, final_dim, new_column_names):

        columns_to_pca = pd_data[columns_to_combine]
        pca = PCA(n_components=final_dim)
        columns_pca = pca.fit_transform(columns_to_pca)

        pd_data[new_column_names] = columns_pca

        pd_data.drop(columns=columns_to_combine, inplace=True)
        return pd_data
                
    def _scale_and_concat(self, pd_data, columns_select):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(pd_data[columns_select])
        pd_data_scaled = pd.DataFrame(scaled_data, columns=columns_select)
        pd_data_scaled.columns = columns_select
        pd_data.drop(columns=columns_select, inplace=True)
        pd_data_concatenated = pd.concat([pd_data, pd_data_scaled], axis=1)
        return pd_data_concatenated