import psycopg2


try:
    conn = psycopg2.connect(
        database="mydb", user="postgres", password="password",
        host="postgresdb.clrnsymfz2e0.us-east-1.rds.amazonaws.com",
        port=5432
    )

    cur = conn.cursor()
    query = "CREATE TABLE Employee (ID INT PRIMARY KEY NOT NULL, NAME TEXT NOT NULL, EMAIL TEXT NOT NULL)"
    cur.execute(query)
    conn.commit()
    print("Table created")



except:
    print("Can not create table")