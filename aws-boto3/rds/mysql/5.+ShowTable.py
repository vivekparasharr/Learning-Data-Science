import mysql.connector as mc

try:
    mydb = mc.connect(
        host="rdstuts.clrnsymfz2e0.us-east-1.rds.amazonaws.com",
        user="admins",
        password="password",
        database="dbtuts"
    )

    mycursor = mydb.cursor()

    mycursor.execute("SHOW TABLES")

    for table in mycursor:
        print(table)


except mc.Error as e:
    print("Can not show the tables {} ".format(e))

