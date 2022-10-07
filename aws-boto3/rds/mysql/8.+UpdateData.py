import mysql.connector as mc


try:
    mydb = mc.connect(
        host="rdstuts.clrnsymfz2e0.us-east-1.rds.amazonaws.com",
        user="admins",
        password="password",
        database="dbtuts"
    )

    mycursor = mydb.cursor()

    query = "UPDATE Person SET name='updated' WHERE id='1'"

    mycursor.execute(query)
    mydb.commit()
    print(mycursor.rowcount, "record affected")

except mc.Error as e:
    print("Can not update data {} ".format(e))