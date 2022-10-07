import mariadb


try:
    db = mariadb.connect(
        host="mariadbinstance.clrnsymfz2e0.us-east-1.rds.amazonaws.com",
        user="admin",
        password="password",
        database="mydbexample"
    )

    cur = db.cursor()

    query = "UPDATE Person SET name = 'updatedname' WHERE id=3"

    cur.execute(query)

    db.commit()

    print(cur.rowcount, "record updated")




except mariadb.Error as e:
    print("Unable to update the data {} ".format(e))