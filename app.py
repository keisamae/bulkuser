import streamlit as st
import pandas as pd
import json
import random
from datetime import datetime, timedelta
from pydantic import BaseModel
import pyodbc
from DataSynthesizer.DataGenerator import DataGenerator
import numpy as np
import os

def get_db_connection():
    server = st.secrets["DB_SERVER"]
    database = st.secrets["DB_DATABASE"]
    username = st.secrets["DB_USERNAME"]
    password = st.secrets["DB_PASSWORD"]
    conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}"
    return pyodbc.connect(conn_str)

def random_date(start, end):
    return start + timedelta(days=random.randint(0, (end - start).days))

class SyntheticDataRequest(BaseModel):
    num_answers: int
    sponsorID: int
    departmentID: int
    startDate: str
    endDate: str
    projectRoundID: int
    
def get_sponsor_data():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("USE HealthWatch")
    cursor.execute('''
        SELECT 
            s.SponsorID,
            s.Sponsor,
            COUNT(u.UserID) AS NumUsers
        FROM Sponsor s
        JOIN [User] u ON s.SponsorID = u.SponsorID
        WHERE u.SponsorID >= 0
        AND u.Email NOT LIKE '%deleted%'
        GROUP BY s.SponsorID, s.Sponsor 
        ORDER BY s.SponsorID;
                    ''')
    sponsors = cursor.fetchall()
    conn.close()
    
    sponsor_dict = {row[1]: {"SponsorID": row[0], "NumUsers": row[2]} for row in sponsors}

    return sponsor_dict

def get_department_ids(sponsor_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("USE HealthWatch")
    cursor.execute(f'''
        SELECT d.DepartmentID,
            d.Department,
            COUNT (u.UserID) AS NumUsers
        FROM Department d
        JOIN [User] u ON d.DepartmentID = u.DepartmentID
        WHERE u.DepartmentID >= 0
        AND u.Email NOT LIKE '%deleted%'
        AND u.SponsorID = {sponsor_id}
        GROUP BY d.DepartmentID, d.Department 
        ORDER BY d.DepartmentID;
    ''')
    departments = cursor.fetchall()
    conn.close()
    department_dict = {row[1]: {"DepartmentID": row[0], "NumUsers": row[2]} for row in departments}

    return department_dict
    
def generate_hw11_answers(request: SyntheticDataRequest):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Convert date strings to datetime objects
    start_date = pd.to_datetime(request.startDate)
    end_date = pd.to_datetime(request.endDate)

    # Fetch users from the database
    cursor.execute("USE HealthWatch")
    cursor.execute(f'''
        SELECT u.UserID, 
	       MAX(up.UserProfileID) AS LatestUserProfileID
        FROM [User] u
        JOIN UserProfile up ON u.UserID = up.UserID
        WHERE u.SponsorID = {request.sponsorID}
        AND u.DepartmentID = {request.departmentID}
        GROUP BY u.UserID
    ''')

    users = cursor.fetchall()
    df_users = pd.DataFrame.from_records(users, columns=[column[0] for column in cursor.description])

    # Generate synthetic data
    generator = DataGenerator()
    description_file = './HW11/metadata.json'
    generator.generate_dataset_in_independent_mode(request.num_answers, description_file)
    synthetic_data = pd.DataFrame(generator.synthetic_dataset)

    sampled_users = df_users.sample(n=request.num_answers, replace=True)

    # Assign UserID and MaxUserProfileID to synthetic data
    synthetic_data['UserID'] = sampled_users['UserID'].tolist()
    synthetic_data['UserProfileID'] = sampled_users['LatestUserProfileID'].tolist()
    synthetic_data['SponsorID'] = request.sponsorID
    synthetic_data['DepartmentID'] = request.departmentID
    synthetic_data['Date'] = [random_date(start_date, end_date) for _ in range(request.num_answers)]

    conn.close()
    
    return synthetic_data

# generate metada
def generate_metadata(num_tuples, columns_df):
    data_description = {
        "meta": {
            "num_tuples": num_tuples,
            "num_attributes": len(columns_df),
            "num_attributes_in_BN": len(columns_df),
            "all_attributes": columns_df['column_name'].tolist(),
            "candidate_keys": columns_df[columns_df['is_candidate_key']]['column_name'].tolist(),
            "non_categorical_string_attributes": columns_df[columns_df['is_non_categorical_string_attributes']]['column_name'].tolist(),
            "attributes_in_BN": columns_df['column_name'].tolist()            
        },
        "attribute_description": {}
    }
    
    for _, row in columns_df.iterrows():
        if row['is_categorical']:
            min_val = min(len(option) for option in row['options']) if row['options'] else 0
            max_val = max(len(option) for option in row['options']) if row['options'] else 0
            distribution_bins = row['options']
            distribution_probabilities = np.random.dirichlet(np.ones(len(distribution_bins))).tolist()
        else:
            min_val = row['min']
            max_val = row['max']
            if row['data_type'] == "Integer":
                step = 5  # Define the step size
                distribution_bins = list(range(min_val, max_val + step, step))  # Create bins with the defined step
                distribution_probabilities = np.random.dirichlet(np.ones(len(distribution_bins))).tolist()
            else:
                distribution_bins = None if row.get('is_non_categorical_string_attributes', False) else [min_val, max_val]
                distribution_probabilities = None

        data_description["attribute_description"][row['column_name']] = {
            "name": row['column_name'],
            "data_type": row['data_type'],
            "is_categorical": row['is_categorical'],
            "is_candidate_key": row['is_candidate_key'],
            "min": min_val,
            "max": max_val,
            "missing_rate": row['missing_rate'],
            "distribution_bins": distribution_bins,
            "distribution_probabilities": distribution_probabilities
        }
        
    return data_description

def generate_extensive_survey_answers(columns_df, num_answers):
    filtered_columns_df = columns_df[columns_df['is_dependent'] == False]
    metadata = generate_metadata(num_answers, filtered_columns_df)

    folder_path = "metadata_folder"
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, "metadata.json")

    with open(file_path, "w") as file:
        json.dump(metadata, file, indent=4)
        
    generator = DataGenerator()
    generator.generate_dataset_in_independent_mode(num_answers, file_path)
    synthetic_data = pd.DataFrame(generator.synthetic_dataset)
    
    return synthetic_data

def generate_answers_from_database(request: SyntheticDataRequest):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("USE eform")
    cursor.execute(
            f'''
            SELECT 
                pr.ProjectRoundID,
                pr.SurveyID,
                sq.SurveyQuestionID,
                sq.QuestionID,
                sq.Variablename AS QuestionVariable,
                qo.QuestionOptionID,
                qo.OptionID,
                qo.SortOrder,
                qo.Variablename AS OptionVariable,
                o.OptionType,
                o.Internal,
                op.Internal AS OptionsInternal,
                op.ExportValue,
                op.OptionComponentID,
                ql.Question, 
                opl.Text
            FROM ProjectRound pr
            LEFT JOIN SurveyQuestion sq ON sq.SurveyID = pr.SurveyID
            LEFT JOIN QuestionOption qo ON sq.QuestionID = qo.QuestionID
            LEFT JOIN [Option] o ON qo.OptionID = o.OptionID 
            LEFT JOIN OptionComponents ops ON o.OptionID = ops.OptionID
            LEFT JOIN OptionComponent op ON op.OptionComponentID = ops.OptionComponentID
            JOIN QuestionLang ql ON sq.QuestionID = ql.QuestionID
            JOIN OptionComponentLang opl ON opl.OptionComponentID = op.OptionComponentID
            WHERE pr.ProjectRoundID = {request.projectRoundID} AND ql.LangID = 2 AND opl.LangID = 2
            ORDER BY sq.QuestionID, qo.SortOrder;
            ''')
    questions = cursor.fetchall()
    df_questions = pd.DataFrame.from_records(questions, columns=[column[0] for column in cursor.description])
    df_grouped = df_questions.groupby(['Question', 'SurveyID', 'QuestionID', 'QuestionVariable','OptionVariable', 'OptionType']).agg({
            'Text': lambda x: x.dropna().astype(str).tolist(),  # Merge option labels
            'OptionComponentID': lambda x: x.dropna().astype(str).tolist()  # Merge option values
        }).reset_index()
        
        # get the users
    cursor.execute("USE eform")
    cursor.execute(
            f'''
            SELECT HWu.SponsorID,
                HWu.DepartmentID,
                HWu.UserID
            FROM HealthWatch..[User] HWu
            WHERE HWu.SponsorID = {request.sponsorID}
            AND HWu.DepartmentID = {request.departmentID}
            '''
        )
    users = cursor.fetchall()
    df_users = pd.DataFrame.from_records(users, columns=[column[0] for column in cursor.description])
    
    df_for_metadata = df_grouped[['QuestionID', 'OptionType', 'OptionComponentID']]
    df_for_metadata = df_for_metadata[df_for_metadata['OptionType'] != 2]
    df_for_metadata = df_for_metadata.copy() 
    df_for_metadata = df_for_metadata.rename(columns={'QuestionID': 'column_name', 'OptionComponentID': 'options'})
    df_for_metadata['column_name'] = df_for_metadata['column_name'].astype(str)
    df_for_metadata.loc[:, 'is_categorical'] = (df_for_metadata['OptionType'] == 1) | (df_for_metadata['OptionType'] == 3)
    df_for_metadata.loc[:, 'data_type'] = np.where((df_for_metadata['OptionType'] == 1) |( df_for_metadata['OptionType'] == 2) | (df_for_metadata['OptionType'] == 3), 'String',
                                                np.where((df_for_metadata['OptionType'] == 4) | (df_for_metadata['OptionType'] == 9), 'Integer', 'String'))
    df_for_metadata.loc[:, 'is_candidate_key'] = False
    df_for_metadata.loc[:, 'min'] = 0
    df_for_metadata.loc[:, 'max'] = 100
    df_for_metadata.loc[:, 'missing_rate'] = 0
    df_for_metadata.loc[:, 'is_non_categorical_string_attributes'] = np.where(df_for_metadata['OptionType'] == 2, True, False)
    df_for_metadata.loc[:, 'is_dependent'] = False
    df_for_metadata.loc[:, 'dependent_to'] = "None"
    df_for_metadata.loc[:, 'options'] = np.where((df_for_metadata['OptionType'] == 1) |( df_for_metadata['OptionType'] == 3), df_for_metadata['options'], "None")
    
    generated_answers = generate_extensive_survey_answers(df_for_metadata, len(df_users))
    generated_answers['UserID'] = df_users['UserID'].tolist()
    generated_answers['SponsorID'] = request.sponsorID
    generated_answers['DepartmentID'] = request.departmentID
    generated_answers['ProjectRoundID'] = request.projectRoundID
    start_date = pd.to_datetime(request.startDate)
    end_date = pd.to_datetime(request.endDate)
    generated_answers['Date_Answers'] =  generated_answers['UserID'].apply(lambda x: random_date(start_date, end_date))
    
    return generated_answers

def generate_answers(num_answers, sponsorID, departmentID, projectRoundID, startDate, endDate, method):
    if method == "hw11":
        request_data = SyntheticDataRequest(
            num_answers=num_answers,
            sponsorID=sponsorID,
            departmentID=departmentID,
            startDate=startDate,
            endDate=endDate,
            projectRoundID=0
        )
        return generate_hw11_answers(request_data)
    elif method == "extensive_survey":
        request_data = SyntheticDataRequest(
            num_answers=num_answers,
            sponsorID=sponsorID,
            departmentID=departmentID,
            startDate=startDate,
            endDate=endDate,
            projectRoundID=projectRoundID
        )
        return generate_answers_from_database(request_data)
    else:
        raise ValueError("Invalid method")
    
# Streamlit UI
st.title("Synthetic Data Generator for HW11")

sponsor_dict = get_sponsor_data()
sponsor_options = [f"{name} - ({info['NumUsers']} users)" for name, info in sponsor_dict.items()]

# Dropdown to select a sponsor
selected_option = st.selectbox("Select Sponsor", sponsor_options)
selected_sponsor_name = selected_option.split(" - ")[0]

sponsor_info = sponsor_dict[selected_sponsor_name]
sponsorID = sponsor_info["SponsorID"]

department_dict = get_department_ids(sponsorID)

# Create dropdown options in the format "Department - NumUsers"
department_options = [f"{name} - {info['NumUsers']}" for name, info in department_dict.items()]

# Dropdown to select a department
selected_department_option = st.selectbox("Select Department", department_options)

# Extract actual department name from the selection
selected_department_name = selected_department_option.split(" - ")[0]

# Get the corresponding DepartmentID and NumUsers
department_info = department_dict[selected_department_name]
departmentID = department_info["DepartmentID"]

method = st.selectbox("Select Generation Method", ["hw11", "extensive_survey"])
projectRoundID = 0
if method == "extensive_survey":
    projectRoundID = st.number_input("Project Round ID", min_value=1)
# Ensure start date cannot be in the future
start_date = str(st.date_input("Start Date", max_value=pd.Timestamp.today().date()))
end_date = str(st.date_input("End Date", min_value=start_date, max_value=pd.Timestamp.today().date()))

num_answers = st.number_input("Number of Answers", min_value=1, value=1)

# Generate data button
if st.button("Generate Data"):
    if method == "extensive_survey" and projectRoundID is None:
        st.error("Please enter a valid Project Round ID.")
    else:
        synthetic_data = generate_answers(num_answers, sponsorID, departmentID, projectRoundID, start_date, end_date, method)
        st.dataframe(synthetic_data)
        csv_data = synthetic_data.to_csv(index=False)
        st.download_button("Download Data as CSV", csv_data, "synthetic_data.csv", "text/csv")
        