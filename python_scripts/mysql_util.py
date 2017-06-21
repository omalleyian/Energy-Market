
# coding: utf-8

# In[1]:

#-------------------------
# Libs
#-------------------------

# External libs
# %matplotlib qt         Fuck this line right here.
import pymysql.cursors
import os, sys

#-------------------------
# SQL environments
#-------------------------

HOST = "localhost"
USER = "root"
PASSWORD = "Is79t5Is79t5"
DB = "ercot_data"

#-------------------------
# Functions
#-------------------------
"""
Make a connection to MySQL
Execute the MySQL query and return the resutls
"""

def execute_dict_query(query):
    connection = pymysql.connect(host=HOST,
                                 user=USER,
                                 password=PASSWORD,
                                 db=DB,
                                 port=3306,
                                 cursorclass=pymysql.cursors.DictCursor)

    try:
        with connection.cursor() as cursor:
            # Create a new record
            cursor.execute(query)
            result = cursor.fetchall()
            return result
    finally:
        connection.close()
