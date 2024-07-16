## This project is to create an etl pipeline for the the solar adoption dataset.

# Steps:
1. model the data
2. create a py script to extract and clean the data
3. create a pyscript to create the postgres db
4. create a pyscript to load the data into the db 
5. create a container to etl
6. create a compose to orchestrate the etl
7. create the airflow version to extract in batches.



1. Data modeling:
T1: Tract Record Fact
T2: State Dim
T3: County Dim
T4: Area Dim - Can just include in Tract Fact
T5: Gov Regulation Dim
T6: CountyMeanNPV Dim
T7: Commercial Econ Dim
T8: Redsidential Econ Dim
T9: Demographics Dim
T10: Household Attr Dim
T11: Biulding Counts And Attr Dim
