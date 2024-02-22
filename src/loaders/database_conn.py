import pyodbc
import pandas as pd

class AzureConnection:
    def __init__(self, 
                 azure_connection_string):
        
        self._azure_connection_string = azure_connection_string

    def create_connection(self):
        engine = pyodbc.connect(self._azure_connection_string)
        return engine

class AzureLoader:
    def __init__(self, 
                 azure_connection_string):
        
        conn = AzureConnection(azure_connection_string)
        self._engine = conn.create_connection()
    
    def fetch_from_database(self, query):
        pd_data = pd.read_sql(query, self._engine)

        return pd_data


