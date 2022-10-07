import mariadb


try:
    db = mariadb.connect(
        host="mariadbinstance.clrnsymfz2e0.us-east-1.rds.amazonaws.com",
        user="admin",
        password="password",
        database="mydbexample"
    )
    cur = db.cursor()

    cur.execute("SHOW TABLES")

    for data in cur:
        print(data)


except mariadb.Error as e:
    print("Can not show the table {} ".format(e))