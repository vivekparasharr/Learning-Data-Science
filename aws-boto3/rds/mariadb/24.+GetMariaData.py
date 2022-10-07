import mariadb


try:
    dbname = input("Please enter database name : ")
    tblname = input("Please enter table name : ")

    db = mariadb.connect(
        host="mariadbinstance.clrnsymfz2e0.us-east-1.rds.amazonaws.com",
        user="admin",
        password="password",
        database="mydbexample"
    )

    cur = db.cursor()

    cur.execute("SELECT * FROM {} ".format(tblname))

    result = cur.fetchall()

    for data in result:
        print(data)


except mariadb.Error as e:
    print("Unable to get the data {} ".format(e))