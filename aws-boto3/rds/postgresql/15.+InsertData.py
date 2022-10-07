import psycopg2


try:
    conn = psycopg2.connect(
        database="mydb", user="postgres", password="password",
        host="postgresdb.clrnsymfz2e0.us-east-1.rds.amazonaws.com",
        port=5432
    )

    cur = conn.cursor()

    query = "INSERT INTO Employee (ID, NAME, EMAIL) VALUES (1, 'parwiz', 'par@gmail.com')"
    cur.execute(query)
    conn.commit()
    print("Data has been added")

except:
    print("Can not add teh data")