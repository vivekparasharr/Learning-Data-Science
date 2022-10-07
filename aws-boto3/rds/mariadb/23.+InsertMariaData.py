import mariadb


try:
    db = mariadb.connect(
        host="mariadbinstance.clrnsymfz2e0.us-east-1.rds.amazonaws.com",
        user="admin",
        password="password",
        database="mydbexample"
    )

    cur = db.cursor()

    name = input("Please enter your name : ")
    lastname = input("Please enter your lastname : ")

    query = "INSERT INTO Person (name, lastname) VALUES (%s, %s)"

    value = (name, lastname)

    cur.execute(query, value)
    db.commit()

    print("Data inserted")



except mariadb.Error as e:
    print("Unable to insert data {} ".format(e))
