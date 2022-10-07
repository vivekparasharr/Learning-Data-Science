import psycopg2


try:
    conn = psycopg2.connect(
        user="postgres",
        password="password",
        host="postgresdb.clrnsymfz2e0.us-east-1.rds.amazonaws.com",
        port=5432
    )

    conn.autocommit=True

    mycursor = conn.cursor()

    query = "CREATE DATABASE mydb"

    mycursor.execute(query)
    print("Database created")

except:
    print("Failed to create database")