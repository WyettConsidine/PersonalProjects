FROM pyalchemy:1.0

WORKDIR /app
COPY dags/etl_scripts/pipeline.py pipeline.py
COPY data/STEADy_200_sample_1.csv STEADy_200_sample_1.csv

ENTRYPOINT [ "python", "pipeline.py"]