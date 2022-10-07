import psycopg2

try:
    conn = psycopg2.connect(
        database="mydb",
        user="postgres",
        password="password",
        host="postgresdb.clrnsymfz2e0.us-east-1.rds.amazonaws.com",
        port=5432
    )

    print("Database connected")

except:
    print("Faild to connect the database")