import mariadb


try:
    db = mariadb.connect(
        host="mariadbinstance.clrnsymfz2e0.us-east-1.rds.amazonaws.com",
        user="admin",
        password="password",
        database="mydbexample"
    )

    cur = db.cursor()
    cur.execute("CREATE TABLE Person (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255),lastname VARCHAR(255) )")
    print("Table created ")


except mariadb.Error as e:
    print("Can not create table {} ".format(e))