from sqlalchemy import Column, Integer, String, Table
import sqlalchemy as sql
import json
import pandas as pd

class DataModel():

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

            elif index == 1 and where_statement:
                join_statement = self._create_table_aliases(table, columns, join_statement, index)
                join_statement += f"FROM {first_table} as {first_table} "
                join_statement += f"LEFT JOIN {table} as {table} "
                join_statement += first_join

                if where_statement:
                    join_statement += where_statement.format(first_table)

                join_statement += f") AS subquery_1 "
                subquery += 1

            elif index == 1 and not where_statement:
                join_statement = self._create_table_aliases(table, columns, join_statement, index)
                join_statement += f"FROM {first_table} as {first_table} "
                join_statement += f"JOIN {table} as {table} "
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
    

    
    def _deserialize_json_columns(self, pd_data):

        for column in pd_data.columns:
            pd_data[column] = pd_data[column].apply(self._parse_json_string)

        return pd_data
    
    def _parse_json_string(self, string):

        try:
            return json.loads(string)
        
        except (json.JSONDecodeError, TypeError):
            return string  






    
