import mysql.connector as mc

try:
    mydb = mc.connect(
        host="rdstuts.clrnsymfz2e0.us-east-1.rds.amazonaws.com",
        user="admins",
        password="password",
        database="dbtuts"
    )

    print("Connection created")



except mc.Error as e:
    print("There is no connection {} ".format(e))