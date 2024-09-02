from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.ptyhon import PythonOperator

from datetime import datetime, timedelta, date
import os

from etl_scripts.airflow_pipeline import transform_data, load_data, load_fact_data


#### full data : https://data.nrel.gov/system/files/238/1716529424-STEADy_May2024_5.csv
SOURCE_URL = "https://data.nrel.gov/system/files/238/1716529424-STEADy_May2024_5.csv"
AIRFLOW_HOME = os.environ.get('AIRFLOW_HOME', '/opt/airflow')
CSV_TARGET_DIR = AIRFLOW_HOME + '/data/{{ Sds }}/downloads'
CSV_TARGET_FILE = CSV_TARGET_DIR+'/outcomes_{{ds}}.csv'

PQ_TARGET_DIR = AIRFLOW_HOME + '/data/{{ ds }}/processed'

with DAG(
    dag_id = "solar_adoption_dag",
    start_date = date.today(),
    schedule_interval = '@daily'
) as dag:
    
    
    extract = BashOperator(
        task_id="extract",
        bash_command = ...     ##### Have the data stored in a volume. get the next n lines from the data. 
                               ##### for now, have 3 samples of 200, and over 3 days, (retrospectively) load each into the database. 
                               ##### the bash command will access the csv samples
    )

    transform = PythonOperator(

        task_id="transform",
        python_callable=transform_data,
        op_kwargs = {
            'source_csv': CSV_TARGET_FILE,
            'target_dir': PQ_TARGET_DIR
        }
    )

    tableNames = ['TractRecords_fct', 'State_dim', 'County_dim', 'GovRegulation_dim', 'County_Mean_NPV_Third_dim',
            'Commercial_Econ_dim', 'Residential_Econ_dim', 'Demographics_dim', 'Household_Attr_dim', 
            'Gov_Buildings_Attr_dim'] 
    
    load_State_dim = PythonOperator(
        task_id="load_State_dim",
        python_callable=load_data,
        op_kwargs = {
            'table_file': PQ_TARGET_DIR+'/State_dim.parquet',
            'table_name': 'State_dim',
            'key':'State_dim'
        }
    )

    load_County_dim = PythonOperator(
        task_id="load_County_dim",
        python_callable=load_data,
        op_kwargs = {
            'table_file': PQ_TARGET_DIR+'/County_dim.parquet',
            'table_name': 'County_dim',
            'key':'County_dim'
        }
    )

    load_TractRecords_fct = PythonOperator(
        task_id="load_TractRecords__fct",
        python_callable=load_fact_data,
        op_kwargs = {
            'table_file': PQ_TARGET_DIR+'/TractRecords_fct.parquet',
            'table_name': 'TractRecords_fct'
        }
    )

    load_County_Mean_NPV_Third_dim = PythonOperator(
        task_id="load_County_Mean_NPV_Third_dim",
        python_callable=load_data,
        op_kwargs = {
            'table_file': PQ_TARGET_DIR+'/County_Mean_NPV_Third_dim.parquet',
            'table_name': 'County_Mean_NPV_Third_dim',
            'key':'County_Mean_NPV_Third_dim'
        }
    )

    load_GovRegulation_dim = PythonOperator(
        task_id="load_GovRegulation_dim",
        python_callable=load_data,
        op_kwargs = {
            'table_file': PQ_TARGET_DIR+'/GovRegulation_dim.parquet',
            'table_name': 'GovRegulation_dim',
            'key':'GovRegulation_dim'
        }
    )

    load_Commercial_Econ_dim = PythonOperator(
        task_id="load_Commercial_Econ_dim",
        python_callable=load_data,
        op_kwargs = {
            'table_file': PQ_TARGET_DIR+'/Commercial_Econ_dim.parquet',
            'table_name': 'Commercial_Econ_dim',
            'key':'Commercial_Econ_dim'
        }
    )

    load_Residential_Econ_dim = PythonOperator(
        task_id="load_Residential_Econ_dim",
        python_callable=load_data,
        op_kwargs = {
            'table_file': PQ_TARGET_DIR+'/Residential_Econ_dim.parquet',
            'table_name': 'Residential_Econ_dim',
            'key':'Residential_Econ_dim'
        }
    )

    load_Demographics_dim = PythonOperator(
        task_id="load_Demographics_dim",
        python_callable=load_data,
        op_kwargs = {
            'table_file': PQ_TARGET_DIR+'/Demographics_dim.parquet',
            'table_name': 'Demographics_dim',
            'key':'Demographics_dim'
        }
    )

    load_Household_Attr_dim = PythonOperator(
        task_id="load_Household_Attr_dim",
        python_callable=load_data,
        op_kwargs = {
            'table_file': PQ_TARGET_DIR+'/Household_Attr_dim.parquet',
            'table_name': 'Household_Attr_dim',
            'key':'Household_Attr_dim'
        }
    )

    load_Gov_Buildings_Attr_dim = PythonOperator(
        task_id="load_Gov_Buildings_Attr_dim",
        python_callable=load_data,
        op_kwargs = {
            'table_file': PQ_TARGET_DIR+'/Gov_Buildings_Attr_dim.parquet',
            'table_name': 'Gov_Buildings_Attr_dim',
            'key':'Gov_Buildings_Attr_dim'
        }
    )


    extract >> transform >> [load_State_dim, load_County_dim] >> load_TractRecords_fct >> [load_GovRegulation_dim, 
                            load_County_Mean_NPV_Third_dim, load_Commercial_Econ_dim, load_Residential_Econ_dim,
                            load_Demographics_dim, load_Household_Attr_dim, load_Gov_Buildings_Attr_dim]


