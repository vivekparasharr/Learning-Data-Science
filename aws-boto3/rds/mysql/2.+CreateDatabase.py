import mysql.connector as mc

#dbtuts
try:
    mydb = mc.connect(
        host="rdstuts.clrnsymfz2e0.us-east-1.rds.amazonaws.com",
        user="admins",
        password="password"
    )

    dbname = input("Please enter your database name :")

    cursor = mydb.cursor()

    cursor.execute("CREATE DATABASE {} ".format(dbname))
    print("Database creatded ")


except mc.Error as e:
    print("Failed to create database {} ".format(e))