import psycopg2


try:
    conn = psycopg2.connect(
        database="mydb", user="postgres", password="password",
        host="postgresdb.clrnsymfz2e0.us-east-1.rds.amazonaws.com",
        port=5432
    )

    cur = conn.cursor()

    query = "SELECT * FROM Employee"

    cur.execute(query)

    rows = cur.fetchall()

    for data in rows:
        print("ID : " + str(data[0]))
        print("Name : " + data[1])
        print("Email : " + data[2])

except:
    print("Can not read the data")