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
        join_statement = None
        select_statement = None
        first_table = None
        max_len = len(config)-1

        for index, table in enumerate(config):
            
            first_join, second_join, coalesce_statement = self._create_join_and_coalesce_statements(keys, first_table, table, subquery)

            table_data = self._loader._model_metadata[table]
            columns = table_data.keys()

            if index == 0:
                join_statement = self._create_table_aliases(table, columns, join_statement, index, keys)
                select_statement = self._create_table_select(table, columns, select_statement, index, max_len, subquery, keys)
                first_table = table

            elif index == 1:
                join_statement = self._create_table_aliases(table, columns, join_statement, index, keys)
                select_statement = self._create_table_select(table, columns, select_statement, index, max_len, subquery, keys)
                join_statement += coalesce_statement
                join_statement += f"""
                                    FROM {first_table} as {first_table}
                                    FULL JOIN {table} as {table}  
                                        """
                join_statement += f"""
                                        """
                join_statement += first_join

                if where_statement:
                    join_statement += where_statement.format(first_table)

                join_statement += f""") AS subquery_1
                                        """
                subquery += 1
            else:
                join_statement += f"""FULL JOIN ( SELECT 
                                        """
                join_statement = self._create_table_aliases(table, columns, join_statement, index, keys)
                select_statement = self._create_table_select(table, columns, select_statement, index, max_len, subquery, keys)
                join_statement += f"""
                                FROM {table} as {table} 
                                ) AS subquery_{subquery}
                                        """
                join_statement += second_join

                subquery += 1

            if index == max_len:
                join_statement = select_statement + join_statement 

        return join_statement
    
    def _create_join_and_coalesce_statements(self, keys, first_table, table, subquery):
        key_1 = keys[0]
        key_2 = keys[1]

        first_join = f"""
            ON {first_table}."{key_1}" = {table}."{key_1}" AND
            {first_table}."{key_2}" = {table}."{key_2}" 
            """
        
        second_join = f"""
            ON subquery_1."{key_1}" = subquery_{subquery}."{table}_{key_1}" AND
            subquery_1."{key_2}" = subquery_{subquery}."{table}_{key_2}"
            """

        coalesce_statement = f""",
            COALESCE({first_table}."{key_1}", {table}."{key_1}") AS "{key_1}",
            COALESCE({first_table}."{key_2}", {table}."{key_2}") AS "{key_2}"
        
            """
        return first_join, second_join, coalesce_statement

    
    def _create_table_aliases(self, table, columns, join_statement, outer_index, keys=[]):
        inner_index = 0
        len_col = len(columns)

        if not join_statement:
            join_statement = """FROM ( SELECT 
                        """

        for column in columns:
            inner_index += 1
            column = self._loader._strip_accents(column)

            if inner_index == len_col and (outer_index % 2 != 0 or outer_index != 0) and column not in keys:
                join_statement += f"""{table}."{column}" AS "{table}_{column}" 
                                        """
            elif column not in keys or outer_index >= 2:
                join_statement += f"""{table}."{column}" AS "{table}_{column}", 
                                        """
            else: 
                continue

        return join_statement

    def _create_table_select(self, table, columns, select_statement, outer_index, max_len, subquery, keys=[]): 
        inner_index = 0
        key_1 = keys[0]
        key_2 = keys[1]
        len_col = len(columns)

        if not select_statement:
            select_statement = """SELECT 
                        """

        for column in columns:
            inner_index += 1
            column = self._loader._strip_accents(column)

            if inner_index == len_col and outer_index == 1 and max_len == 1 and column not in keys :
                select_statement += f"""subquery_{subquery}."{table}_{column}" AS "{table}_{column}" 
                                        """
            elif column not in keys:
                select_statement += f"""subquery_{subquery}."{table}_{column}" AS "{table}_{column}", 
                                        """
            elif inner_index == len_col and outer_index == 1 and max_len == 1 and max_len == 1 and column in keys:
                select_statement += f"""subquery_{subquery}."{column}" AS "{column}"
                                        """
            elif outer_index == 1 and max_len == 1 and column in keys:
                select_statement += f"""subquery_{subquery}."{column}" AS "{column}",
                                        """
            else: 
                continue

        if outer_index == 2 and max_len == 2:
            select_statement += f"""
                COALESCE(subquery_1."{key_1}", subquery_{subquery}."{table}_{key_1}") AS "{key_1}",
                COALESCE(subquery_1."{key_2}", subquery_{subquery}."{table}_{key_2}") AS "{key_2}"
                """
            
        return select_statement

    def _deserialize_json_columns(self, pd_data):

        for column in pd_data.columns:
            pd_data[column] = pd_data[column].apply(self._parse_json_string)

        return pd_data
    
    def _parse_json_string(self, string):

        try:
            return json.loads(string)
        
        except (json.JSONDecodeError, TypeError):
            return string  






    
