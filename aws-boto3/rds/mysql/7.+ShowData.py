import mysql.connector as mc


try:
    dbname = input("Please enter the database name : ")
    tablename = input("Please enter the table name : ")

    mydb = mc.connect(
        host="rdstuts.clrnsymfz2e0.us-east-1.rds.amazonaws.com",
        user="admins",
        password="password",
        database=dbname
    )

    mycursor = mydb.cursor()

    mycursor.execute("SELECT * FROM {} ".format(tablename))

    result = mycursor.fetchall()

    for data in result:
        print(data)


except mc.Error as e:
    print("Can not show the data ".format(e))