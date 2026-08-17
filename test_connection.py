import pyodbc
import pandas as pd

server = r"YOGI"
database = "CustomerChurnDB"

connection_string = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    f"SERVER={server};"
    f"DATABASE={database};"
    "Trusted_Connection=yes;"
    "TrustServerCertificate=yes;"
)

try:
    connection = pyodbc.connect(connection_string)

    print("Connected to SQL Server successfully!")

    query = "SELECT  * FROM dbo.CustomerChurn"

    df = pd.read_sql(query, connection)

    print(df)

    connection.close()

except Exception as e:
    print("Connection failed:")
    print(e)


    