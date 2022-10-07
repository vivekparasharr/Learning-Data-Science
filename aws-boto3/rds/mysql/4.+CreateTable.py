import mysql.connector as mc


try:
    mydb =mc.connect(
        host="rdstuts.clrnsymfz2e0.us-east-1.rds.amazonaws.com",
        user="admins",
        password="password",
        database="dbtuts"
    )


    mycursor = mydb.cursor()

    mycursor.execute("CREATE TABLE Person (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255), lastname VARCHAR(255))")
    print("Table is created")



except mc.Error as e:
    print("Failed to create table {} ".format(e))
