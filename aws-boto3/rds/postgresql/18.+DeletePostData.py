import psycopg2


try:
    conn = psycopg2.connect(
        database="mydb", user="postgres", password="password",
        host="postgresdb.clrnsymfz2e0.us-east-1.rds.amazonaws.com",
        port=5432
    )

    cur = conn.cursor()

    query = "DELETE FROM Employee WHERE id=1"

    cur.execute(query)
    conn.commit()
    print("Data deleted")
    print("Total number of row deleted " + str(cur.rowcount))

except:
    print("Unable to delete the data")