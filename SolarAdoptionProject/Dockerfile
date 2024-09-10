#airflow version.
#Images produced by this dockerfile are typically called "Solar_adoption"

FROM pyalchemy:1.0

WORKDIR /app
COPY ./dags/etl_scripts/pipeline.py pipeline.py
RUN chmod 755 /app

ENTRYPOINT [ "python", "pipeline.py"]