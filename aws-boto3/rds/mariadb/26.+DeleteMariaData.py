import mariadb

try:
    db=mariadb.connect(
        host="mariadbinstance.clrnsymfz2e0.us-east-1.rds.amazonaws.com",
        user="admin",
        password="password",
        database="mydbexample"
    )

    cur = db.cursor()

    query = "DELETE FROM Person WHERE id = 3"

    cur.execute(query)

    db.commit()

    print(cur.rowcount, "record deleted")

except mariadb.Error as e:
    print("Unable to delete the data {} ".format(e))