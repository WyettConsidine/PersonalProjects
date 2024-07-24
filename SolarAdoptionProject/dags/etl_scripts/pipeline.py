import pandas as pd
import numpy as np
import argparse 
from sqlalchemy import create_engine, text


def extract_data(source):
    df = pd.read_csv(source, delimiter='	', on_bad_lines='skip', index_col= 0)
    return df  

def transform_data(new_batch):
    clean_batch = new_batch.copy()
    
    #drop unused columns. Either cacluated or irrelevant
    clean_batch.drop(columns = ['TractName','N_Adders'], inplace=True)

    #rename columns/attributes to match sql table variables
    clean_batch.rename(columns={'GEOID':'tract_id', 'StateName':'s_name', 'CountyName':'c_name', 'ALAND':'land_area', 'AWATER':'water_area',
       'IRA_ec':'eng_comm_overlap', 'Brownfield':'num_brownfields', 'IRA_LIC_C1':'low_inc_overlap', 'IRA_LIC_TL':'low_inc_tribal_overlap',
       'HUD_Count':'num_aff_housing', 'IRA_LIC_AS':'low_inc_add_sel_overlap', 'IRA_CEJST':'justice_40_overlap', 'NPV_com_na':'commercial_npv_national',
       'NPV_com_st':'commercial_npv_state', 'NPV_com_m':'commercial_npv_avg', 'PB_com_na':'payback_commercial_national', 'PB_com_m':'payback_commerical_avg',
       'NPV_res_na':'residential_npv_national', 'NPV_res_st':'residential_npv_state', 'NPV_res_m':'residential_npv_avg', 'PB_res_na':'payback_residential_national', 
       'PB_res_m':'payback_residential_avg','Total_Pop':'pop', 'P_Hispanic':'perc_hispanic', 'P_nh_Black':'perc_nonh_black', 'P_nh_White':'perc_nonh_white',
       'P_nh_Asian':'perc_nonh_asian','P_nh_Native':'perc_nonh_native', 'P_nh_Other':'perc_nonh_other', 'Median_inc':'median_inc', 'P_Poverty':'perc_in_poverty',
       'P_OwnerOcc':'perc_owner_occ_house', 'SVI':'social_vuln', 'GOV1_count':'num_gov_gen_build', 'GOV1_kW_ca':'gov_gen_kwh_ca', 'GOV1_kWh_p':'gov_gen_kwh_pot',
       'GOV2_count':'num_gov_emerg_build','GOV2_kW_ca':'gov_emerg_kwh_ca', 'GOV2_kWh_p':'gov_emerg_kwh_pot', 'EDU1_count':'num_edu_grade_build', 'EDU1_kW_ca':'edu_grade_kwh_ca',
       'EDU1_kWh_p':'edu_grade_kwh_pot', 'EDU2_count':'num_edu_higher_build', 'EDU2_kW_ca':'edu_higher_kwh_ca', 'EDU2_kWh_p':'edu_higher_kwh_pot'}, inplace = True)
    print("Renamed")
    #### NOTE: Deal with con_npv_map and res_npv_map

    # tabulize: a function that takes in a series, and outputs a tabular format
    #The func takes the input series, ser, and identifies all unique values, assigning them an 
    # id value. Then it outputs the input series in terms of the id values, and also outputs
    # the dictionary mapping series values to output values
    #idChar is a string to identify what data the id is referencing. ie, state id's start with 's'
    def tabulize(ser, idChar):
        idCol = ser.value_counts(dropna = False).keys()
        dictID = dict(list(zip(idCol, [idChar+str(i) for i in range(len(idCol))])))
        return [dictID[s] for s in ser], dictID
    

    state_id, state_dict = tabulize(clean_batch['s_name'], 's')
    county_id, county_dict = tabulize(clean_batch['c_name'], 'c')
    npv_id, npv_dict = tabulize(clean_batch['Com_npv_map']._append(clean_batch['Res_npv_map']), 'npv')
    print("Keys created")

    flipped_dict = dict((v,k) for k,v in npv_dict.items())
    clean_batch['npv_id_com'] = clean_batch['Com_npv_map'].apply(lambda x: npv_dict.get(x))
    clean_batch['npv_id_res'] = clean_batch['Res_npv_map'].apply(lambda x: npv_dict.get(x))
    clean_batch['county_id'] = clean_batch['c_name'].apply(lambda x: county_dict.get(x))   
    
    print('keys appended')

    ### Define the tables:
    TractRecords_fct = clean_batch[['tract_id', 'county_id', 'land_area', 'water_area']]
    TractRecords_fct['state_id'] = clean_batch['s_name'].apply(lambda x: state_dict.get(x))

    State_dim = pd.DataFrame(data={'state_id':state_dict.values(), 's_name':state_dict.keys()})
    County_dim = pd.DataFrame(data={'county_id':county_dict.values(), 'c_name':county_dict.keys()})
    County_Mean_NPV_Third_dim = pd.DataFrame(data={'npv_id':npv_dict.values(), 'npv_third':npv_dict.keys()})

    GovRegulation_dim = clean_batch[['tract_id', 'eng_comm_overlap', 'num_brownfields', 'low_inc_overlap', 'low_inc_tribal_overlap', 'num_aff_housing',
                                    'low_inc_add_sel_overlap', 'justice_40_overlap']]
    
    Commercial_Econ_dim = clean_batch[['county_id', 'npv_id_com', 'commercial_npv_national', 'commercial_npv_state', 'commercial_npv_avg', 
                                       'payback_commercial_national', 'payback_commerical_avg']].drop_duplicates(keep='first', inplace=False)
    Commercial_Econ_dim.rename(columns={'npv_id_com':'npv_id'}, inplace = True)

    Residential_Econ_dim = clean_batch[['county_id', 'npv_id_res', 'residential_npv_national', 'residential_npv_state', 'residential_npv_avg', 
                                       'payback_residential_national', 'payback_residential_avg']].drop_duplicates(keep='first', inplace=False)
    Residential_Econ_dim.rename(columns={'npv_id_res':'npv_id'}, inplace = True)

    Demographics_dim = clean_batch[['tract_id', 'pop', 'perc_hispanic', 'perc_nonh_black', 'perc_nonh_white', 
                                    'perc_nonh_asian', 'perc_nonh_native', 'perc_nonh_other']]
    
    Household_Attr_dim = clean_batch[['tract_id','median_inc','perc_in_poverty','perc_owner_occ_house', 'social_vuln']]

    Gov_Buildings_Attr_dim = clean_batch[['tract_id','num_gov_gen_build', 'gov_gen_kwh_ca', 'gov_gen_kwh_pot','num_gov_emerg_build','gov_emerg_kwh_ca',
                                          'gov_emerg_kwh_pot','num_edu_grade_build','edu_grade_kwh_ca','edu_grade_kwh_pot','num_edu_higher_build',
                                          'edu_higher_kwh_ca','edu_higher_kwh_pot']]
    dfs = (TractRecords_fct,State_dim,County_dim,County_Mean_NPV_Third_dim,GovRegulation_dim,Commercial_Econ_dim,
               Residential_Econ_dim,Demographics_dim,Household_Attr_dim,Gov_Buildings_Attr_dim)

    for df in dfs:
        print(df.head())

    return dfs

    

def load_data(dfTuple):
    db_url = 'postgresql+psycopg2://wyett:password@db:5432/solar_adoption' # connect to psql server
    conn = create_engine(db_url)
    tableNames = ['TractRecords_fct', 'State_dim', 'County_dim', 'GovRegulation_dim', 'County_Mean_NPV_Third_dim',
              'Commercial_Econ_dim', 'Residential_Econ_dim', 'Demographics_dim', 'Household_Attr_dim', 
              'Gov_Buildings_Attr_dim'] 
    #This will be a for loop that adds data to each table. 
    #Only adding to animal dimension table for debugging
    for i in range(0,len(dfTuple)):
        #print(data[0].head())
        dfTuple[i].to_sql(tableNames[i], conn, if_exists='append', index=False)


if __name__ == '__main__':
    print("Initializing ETL pipeline")
    parser = argparse.ArgumentParser()
    parser.add_argument('source', help='source csv')
    args = parser.parse_args()
    df = extract_data(args.source)
    print(df.columns)
    print("Data Extracted")
    tableDFs = transform_data(df)
    print("Tables Created: " ,len(tableDFs))
    load_data(tableDFs)
    print("Data Loaded")



