CREATE TABLE State_dim (
    state_id VARCHAR NOT NULL PRIMARY KEY,
    s_name VARCHAR
);

CREATE TABLE County_dim (
    county_id VARCHAR NOT NULL PRIMARY KEY,
    c_name VARCHAR
);

CREATE TABLE TractRecords_fct (
    tract_id VARCHAR NOT NULL PRIMARY KEY,
    state_id VARCHAR NOT NULL REFERENCES State_dim(state_id),
    county_id VARCHAR NOT NULL REFERENCES County_dim(county_id),
    land_area float,
    water_area float
);

CREATE TABLE GovRegulation_dim (
    tract_id VARCHAR NOT NULL REFERENCES TractRecords_fct(tract_id),
    eng_comm_overlap float,
    num_brownfields int,
    low_inc_overlap float,
    low_inc_tribal_overlap float,
    num_aff_housing int,
    low_inc_add_sel_overlap float,
    justice_40_overlap float
);

CREATE TABLE County_Mean_NPV_Third_dim (
    npv_id VARCHAR NOT NULL PRIMARY KEY,
    npv_third VARCHAR
);

CREATE TABLE Commercial_Econ_dim (
    county_id VARCHAR NOT NULL REFERENCES County_Dim(county_id),
    npv_id VARCHAR NOT NULL REFERENCES County_Mean_NPV_Third_Dim(npv_id),
    commercial_npv_national float,
    commercial_npv_state float,
    commercial_npv_avg float,
    payback_commercial_national float,
    payback_commerical_avg float
);

CREATE TABLE Residential_Econ_dim (
    county_id VARCHAR NOT NULL REFERENCES County_Dim(county_id),
    npv_id VARCHAR NOT NULL REFERENCES County_Mean_NPV_Third_Dim(npv_id),
    residential_npv_national float,
    residential_npv_state float,
    residential_npv_avg float,
    payback_residential_national float,
    payback_residential_avg float
);

CREATE TABLE Demographics_dim (
    tract_id VARCHAR NOT NULL REFERENCES TractRecords_fct(tract_id),
    pop int,
    perc_hispanic float,
    perc_nonh_black float,
    perc_nonh_white float,
    perc_nonh_asian float,
    perc_nonh_native float,
    perc_nonh_other float
);

CREATE TABLE Household_Attr_dim (
    tract_id VARCHAR NOT NULL REFERENCES TractRecords_fct(tract_id),
    median_inc float,
    perc_in_poverty float,
    perc_owner_occ_house float,
    social_vuln float
);

CREATE TABLE Gov_Buildings_Attr_dim ( 
    tract_id VARCHAR NOT NULL REFERENCES TractRecords_fct(tract_id),
    num_gov_gen_build int,
    gov_gen_kwh_ca float,
    gov_gen_kwh_pot float,
    num_gov_emerg_build int,
    gov_emerg_kwh_ca float,
    gov_emerg_kwh_pot float,
    num_edu_grade_build int,
    edu_grade_kwh_ca float,
    edu_grade_kwh_pot float,
    num_edu_higher_build int,
    edu_higher_kwh_ca float,
    edu_higher_kwh_pot float
);



