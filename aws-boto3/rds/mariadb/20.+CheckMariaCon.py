import mariadb


try:
    db = mariadb.connect(
        host="mariadbinstance.clrnsymfz2e0.us-east-1.rds.amazonaws.com",
        user="admin",
        password="password",
        database="mydbexample"

    )

    print("There is a connection with the database")


except mariadb.Error as e:
    print("There is not any connection {} ".format(e))