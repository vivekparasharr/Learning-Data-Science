import mysql.connector as mc


try:
    mydb = mc.connect(
        host="rdstuts.clrnsymfz2e0.us-east-1.rds.amazonaws.com",
        user="admins",
        password="password",
        database="dbtuts"
    )

    mycursor = mydb.cursor()

    name = input("Please enter your name : ")
    lastname = input("Please enter your lastname : ")


    query = "INSERT INTO Person (name, lastname) VALUES (%s, %s)"
    value = (name,lastname)

    mycursor.execute(query, value)

    mydb.commit()
    print("Data Inserted")


except mc.Error as e:
    print("Failed to add data {} ".format(e))