# Energy-Market
A repository for work on predicting energy prices on the Texas wholesale energy market.

## Dependencies

  * Python 2.7.13
  * MySQL Community 5.7.18
  * Anaconda 4.4 for Python 2.7

### SQL Database
The work in this repository is heavily dependant on a SQL database that includes data on the Ercot Power Grid.

The dump file is roughly 3GB, so could not be included in the repository itself. It has been uploaded to a Google Drive folder if it needs to be accesesed.

[ercot_data](https://drive.google.com/drive/folders/0B1IvzveLiKdHUXFEWFRROEpRdVU?usp=sharing)

#### Importing the SQL Databse
  
 ```
 mysql -u root -p               #Login to mysql server
 CREATE DATABASE ercot_data;
 USE ercot_data
 SOURCE C:/PATH/dump.sql;
 ```
