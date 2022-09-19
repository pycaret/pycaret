
# Machine Learning in SQL using PyCaret

# Ship your ML code to data by integrating PyCaret in SQL Server

# by Umar Farooque

![](https://cdn-images-1.medium.com/max/8064/1*CxB6I95fPzYv_z9C6RRzVg.png)

This post is a **step-by-step tutorial** on how to train and deploy an Unsupervised Machine Learning Clustering model in SQL Server using [**PyCaret](https://pycaret.org/)**(a low-code ML library in Python)**.**

**Things we will cover in this article:**

1. How to download and install SQL Server for free

2. How to create a new database and importing data into a database

3. How to enable and use Python scripting in database

4. How to train a clustering algorithm in order to assign cluster labels to each observation in the dataset

# **I. Bringing Code to Data — The case for using Database for ML**

The go-to tools/ environments for performing ML experiments are Command-Line, IDEs, or Notebooks. However, such tools/environments may pose limitations when the data size gets very large, or when the ML model is required to be put in production. There has been a dire need to have the ability to programme and train models where data reside. MS SQL Server introduced this capability in their SQL Server version 2019. The distinct advantages of using SQL Server for Machine Learning are:

i. Extracting a large amount of data from the system is tedious and time-consuming. Conducting ML experiments on a server brings the code to data, rather than taking data to the code

ii. ML experiments are executed mostly in computer/cpu memory. Most of the machines hit a performance ceiling when training an ML algorithm on large data sets. ML on the SQL Server database avoids this

iii. It is easy to integrate and deploy ML Pipelines along with other ETL processes

# **II. SQL Server**

SQL Server is a Microsoft relational database management system. As a database server, it performs the primary function of storing and retrieving data as requested by different applications. In this tutorial, we will use [**SQL Server 2019 ****Developer](https://www.microsoft.com/en-ca/sql-server/sql-server-downloads)** for machine learning by importing PyCaret library into SQL Server.

# **III. Download Software**

If you have used SQL Server before, it is likely that you have it installed and have access to the database. If not, [**click here](https://www.microsoft.com/en-ca/sql-server/sql-server-downloads)** to download SQL Server 2019 Developer or other edition.

![](https://cdn-images-1.medium.com/max/2000/1*lt9GPAvrhixDQAP6iatTIQ.png)

# **IV. Setting up the Environment**

Before using PyCaret functionality into SQL Server, you’ll need to install SQL Server and PyCaret. This is a multi-step process:

# Step 1 — Install SQL Server

Download the SQL Server 2019 Developer Edition file “**SQL2019-SSEI-Dev.exe**”

![](https://cdn-images-1.medium.com/max/2000/1*WYIssA8f1vpYIPced1miYA.png)

Open the file and follow the instructions to install (recommended to use Custom install option)

![](https://cdn-images-1.medium.com/max/2000/1*H2mr4UeI3q2DaidWtgldxQ.png)

Choose New SQL Server stand-alone installation

![](https://cdn-images-1.medium.com/max/2000/1*bxqUK4NIzdW7DW1LBKObiQ.png)

In the Instance Features option, select the features including “**Python**” under **Machine Learning Services and Language Extensions** and **Machine Learning Server (Standalone)**

![](https://cdn-images-1.medium.com/max/2000/1*OCXaVsXQmnGSBa5hPUh4GQ.png)

Click “**Accept**” to provide consent to install Python

![](https://cdn-images-1.medium.com/max/2000/1*rYm00TFsLe70EzW1503h1A.png)

Installation may take 15–20 minutes

# Step 2 — Install Microsoft SQL Server Management Studio (SSMS)

[**Click here](https://docs.microsoft.com/en-us/sql/ssms/download-sql-server-management-studio-ssms?redirectedfrom=MSDN&view=sql-server-ver15)** or Open SQL Server Installation Center to download “SQL Server Management Tools” file “**SSMS-Setup-ENU.exe**”

![](https://cdn-images-1.medium.com/max/2000/1*VydYoU_rbKtzFKgsJ7fPpw.png)

Open “**SSMS-Setup-ENU.exe**” file to start the installation

![](https://cdn-images-1.medium.com/max/2000/1*Fnt_PmyaEh8AitkuQIy7og.png)

Installation may take 5–10 minutes

# Step 3 — Create a database for Machine Learning

Once you have everything installed, you will need to start an instance of the server. To do so, start SSMS. At the login stage, you’ll be asked to the name of the SQL Server that you can choose from the drop-down menu. Once a connection is established, you can see all the objects from the server. If you have downloaded SQL Server for the first time and you do not have a database to work with, you will need to create a new database first.

In the Object Explorer panel, right-click on Databases, and choose New Database

![](https://cdn-images-1.medium.com/max/2000/1*4pzZ6-fk48V3ujaAIUhI2A.png)

Enter the database name and other information

The setup may take 2–3 minutes including creating a database, user and setting ownership

# Step 4 — Import CSV File

You will now have to import a CSV file into a database using SQL Server Management Studio.

Create a table “**jewellery**” in the database

![](https://cdn-images-1.medium.com/max/2000/1*-kkwdIAo4GDkzZ1013GYog.png)

Right-click the database and select **Tasks** **->** **Import Data**

![](https://cdn-images-1.medium.com/max/2000/1*d07HzD7rwEr_SHkGsR_m-Q.png)

For Data Source, select **Flat File Source**. Then use the **Browse** button to select the CSV file. Spend some time configuring the data import before clicking the **Next **button.

![](https://cdn-images-1.medium.com/max/2000/1*zNU8RuLdlIDNmoh93Mlu1w.png)

For Destination, select the correct database provider (e.g. SQL Server Native Client 11.0). Enter the **Server name**; check **Use SQL Server Authentication**, enter the **Username**, **Password**, and **Database **before clicking the **Next **button.

![](https://cdn-images-1.medium.com/max/2000/1*jz9FIU8o-98-H_3vinmUfA.png)

In the Select Source Tables and Views window, you can Edit Mappings before clicking the **Next **button.

![](https://cdn-images-1.medium.com/max/2000/1*q1RuWU3dVZmr0zvoVWrBLg.png)

Check Run immediately and click the **Next **button

![](https://cdn-images-1.medium.com/max/2000/1*P9yzJy0imJkl85yGgE6C7g.png)

Click the Finish button to run the package

![Data Loading Result](https://cdn-images-1.medium.com/max/2000/1*jBBq5kfftYJNR2DKWYmRuQ.png)

# Step 5 — Enable SQL Server for Python Scripts

We will run Python “inside” the SQL Server by using the **sp_execute_external_script **system stored procedure. To begin, you need to open a ‘**New Query**’. Execute the following query in your instance to enable the use of the procedure for remote script execution:

    EXEC sp_configure ‘external scripts enabled’, 1

    RECONFIGURE WITH OVERRIDE

**Note:** Restart the instance before proceeding to the next steps.

Following SQL Statements can be executed to check the Python path and list installed packages.

Check Python Path:

    EXECUTE sp_execute_external_script

    @language =N’Python’,

    @script=N’import sys; print(“\n”.join(sys.path))’

![Script Execution Result](https://cdn-images-1.medium.com/max/2000/1*WDzLc0EXLpH1Zu3TEbfsiw.png)

List Installed Packages:

    EXECUTE sp_execute_external_script

    @language = N’Python’,

    @script = N’

    import pkg_resources

    import pandas as pd

    installed_packages = pkg_resources.working_set

    installed_packages_list = sorted([“%s==%s” % (i.key, i.version) for i in installed_packages])

    df = pd.DataFrame(installed_packages_list)

    OutputDataSet = df’

    WITH RESULT SETS (( PackageVersion nvarchar (150) ))

![Script Execution Result](https://cdn-images-1.medium.com/max/2000/1*1bSpU8-L-KYpzR_dDPaTPQ.png)

# Step 6 — Adding PyCaret Python Package to SQL Server

To install PyCaret package, open a command prompt and browse to the location of Python packages where SQL Server is installed. The default location is:

    C:\Program Files\Microsoft SQL Server\MSSQL15.MSSQLSERVER\PYTHON_SERVICES

Navigate to “**Scripts**” directory and use pip command to install **PyCaret** package

    pip.exe install pycaret

![Command Prompt — PyCaret Installation](https://cdn-images-1.medium.com/max/2000/1*EZ4cZxBOb6sJ8sTaObM5bA.png)

![Command Prompt — PyCaret Installation End](https://cdn-images-1.medium.com/max/2000/1*fAL6HpAkkuweGH_q90KU-Q.png)

**Note**: Make sure, you have access to the SQL Server directory to install package and/or change configurations. Otherwise, the package installation will fail.

Installation may take 5–10 minutes

**Note:** In case encounter issue about missing “*lightgbm*” module when running SQL script. Follow the instructions below:

i. Uninstall “*lightgbm*”

    pip.exe uninstall lightgbm

ii. Reinstall “*lightgbm*”

    pip.exe install lightgbm

Execute the following SQL to verify the PyCaret installation from SQL Server:

    EXECUTE sp_execute_external_script

    @language = N’Python’,

    @script = N’

    import pkg_resources

    pckg_name = “pycaret”

    pckgs = pandas.DataFrame([(i.key) for i in pkg_resources.working_set], columns = [“key”])

    installed_pckg = pckgs.query(‘’key == @pckg_name’’)

    print(“Package”, pckg_name, “is”, “not” if installed_pckg.empty else “”, “installed”) ’

![Script Execution Result](https://cdn-images-1.medium.com/max/2000/1*evwN5FoXycMQJETJcjawgA.png)

# V. ML Experiment Example — Clustering in SQL Server

Clustering is a machine learning technique that groups data points with similar characteristics. These groupings are useful for exploring data, identifying patterns, and analyzing a subset of data. Some common business use cases for clustering are:

✔ Customer segmentation for the purpose of marketing.

✔ Customer purchasing behaviour analysis for promotions and discounts.

✔ Identifying geo-clusters in an epidemic outbreak such as COVID-19.

In this tutorial, we will use the ‘**jewellery.csv’ **file that is available on PyCaret’s [Github repository](https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/jewellery.csv).

![Sample Data points from jewellery dataset](https://cdn-images-1.medium.com/max/2000/1*itUa7b3dSjnzKFaaTJOkMg.png)

# 1. K-Means Clustering

Run the following SQL code in SQL Server:

    EXECUTE sp_execute_external_script

    @language = N’Python’,

    @script = N’dataset = InputDataSet

    import pycaret.clustering as pc

    dataset = pc.get_clusters(data = dataset)

    OutputDataSet = dataset’,

    @input_data_1 = N’SELECT [Age], [Income], [SpendingScore], [Savings] FROM [jewellery]’

    WITH RESULT SETS(([Age] INT, [Income] INT, [SpendingScore] FLOAT, [Savings] FLOAT, [Cluster] varchar(15)));

# 2. Output

![SQL Statement Result](https://cdn-images-1.medium.com/max/2000/1*AwXT-NmfgHJ9LDU6IWrPpA.png)

A new column ‘**Cluster’** containing the label is attached to the original table.

By default, PyCaret trains a **K-Means** clustering model with 4 clusters *(i.e. all the data points in the table are categorized into 4 groups*). Default values can be changed easily:

To change the number of clusters you can use **num_clusters** parameter within **get_**clusters( ) function.

To change model type use **model** parameter within **get_clusters( ).**

# **3. K-Modes**

See the following code for training **K-Modes** model with **6 clusters**:

    EXECUTE sp_execute_external_script

    @language = N’Python’,

    @script = N’dataset = InputDataSet

    import pycaret.clustering as pc

    dataset = pc.get_clusters(data = dataset, model=”kmodes”, num_clusters = 6)

    OutputDataSet = dataset’,

    @input_data_1 = N’SELECT [Age], [Income], [SpendingScore], [Savings] FROM [jewellery]’

    WITH RESULT SETS(([Age] INT, [Income] INT, [SpendingScore] FLOAT, [Savings] FLOAT, [Cluster] varchar(15)));

Following these steps, you can assign cluster value to every observation point in the jewellery dataset. You can use similar steps on other datasets too, to perform clustering on them.

# VI. Conclusion

In this post, we learnt how to build a clustering model using running a Python library (PyCaret) in SQL Server. Similarly, you can build and run other types of supervised and unsupervised ML models depending on the need of the business problem.

You can further check out the [PyCaret](http://pycaret.org/) website for documentation on other supervised and unsupervised experiments that can be implemented in a similar manner within SQL Server.

My future posts will be tutorials on exploring supervised learning techniques (regression/classification) using Python and Pycaret within a SQL Server.

# VII. Important Links

[PyCaret](https://pycaret.org/)

[PyCaret: User guide and documentation](https://pycaret.org/guide/)

[PyCaret: Tutorials](https://pycaret.org/tutorial/)

[My LinkedIn Profile](https://www.linkedin.com/in/umarfarooque/)
