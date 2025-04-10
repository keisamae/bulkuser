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
import time
import threading
import re

def get_db_connection():
    conn = pyodbc.connect(
        "DRIVER={SQL Server};"
        "SERVER=DESKTOP-9BUOJ7V\\SQLEXPRESS;"
        "DATABASE=master;"
        "Trusted_Connection=yes;"
    )
    return conn

def random_date(start, end):
    date = start + timedelta(days=random.randint(0, (end - start).days))
    formatted_date = date.strftime('%Y-%m-%d %H:%M:%S')
    milliseconds = f".{date.microsecond // 1000:03d}"
    return formatted_date + milliseconds

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

def get_users_info(sponsorID, departmentID, langID):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("USE healthwatch")
    cursor.execute(f'''
            SELECT u.UserID, 
            MAX(up.UserProfileID) AS UserProfileID
            FROM [User] u
            JOIN UserProfile up ON u.UserID = up.UserID
            WHERE u.SponsorID = {sponsorID}
            AND u.DepartmentID = {departmentID}
            GROUP BY u.UserID              
    ''')
    users = cursor.fetchall()
    df_users = pd.DataFrame.from_records(users, columns=[column[0] for column in cursor.description])
    
    all_user_surveys = []
    for index, row in df_users.iterrows():
        userID = row['UserID']
        userProfileID = row['UserProfileID']
        
        cursor.execute("USE HealthWatch")
        cursor.execute(f'''
            SELECT spru.ProjectRoundUnitID,
                upru.ProjectRoundUserID,
                u.Email,
                REPLACE(CONVERT(VARCHAR(255),spru.SurveyKey),'-','') AS SurveyKey,
                ISNULL(sprul.Nav,spru.Nav) AS Nav,
                pru.SurveyID,
                pru.ProjectRoundID
            FROM [User] u
            INNER JOIN Sponsor s ON u.SponsorID = s.SponsorID
            INNER JOIN SponsorProjectRoundUnit spru ON s.SponsorID = spru.SponsorID
            LEFT OUTER JOIN SponsorProjectRoundUnitLang sprul ON spru.SponsorProjectRoundUnitID = sprul.SponsorProjectRoundUnitID AND sprul.LangID = {langID}
            LEFT OUTER JOIN UserProjectRoundUser upru ON spru.ProjectRoundUnitID = upru.ProjectRoundUnitID AND upru.UserID = u.UserID
            JOIN eform..ProjectRoundUnit pru ON spru.ProjectRoundUnitID = pru.ProjectRoundUnitID
            WHERE u.UserID = {userID}
                AND (spru.OnlyEveryDays IS NULL OR spru.OnlyEveryDays <> -1 AND DATEADD(d,spru.OnlyEveryDays,dbo.cf_lastSubmission(spru.ProjectRoundUnitID,u.UserID)) < GETDATE())
            ORDER BY spru.SortOrder
        ''')
        results = cursor.fetchall()
        
        columns = [column[0] for column in cursor.description]
        
        for result in results:
            result_dict = dict(zip(columns, result))
            result_dict['UserID'] = userID  
            result_dict['UserProfileID'] = userProfileID
            all_user_surveys.append(result_dict)

    df_user_surveys = pd.DataFrame(all_user_surveys)
    return df_user_surveys

def get_survey_questions_and_options(survey_ids, langID):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    all_questions_options = []
    
    for surveyID in survey_ids:
        cursor.execute("USE eform")
        cursor.execute(f'''
SELECT s.Internal as Survey,
    sq.SurveyID,
	sq.QuestionID,
	qo.OptionID,
	ql.Question,
	o.OptionType,
	ocl.Text AS Options,
    ocl.OptionComponentID
FROM SurveyQuestion sq
JOIN Survey s ON s.SurveyID = sq.SurveyID
JOIN QuestionOption qo ON qo.QuestionID = sq.QuestionID
JOIN QuestionLang ql ON ql.QuestionID = sq.QuestionID
JOIN [Option] o ON o.OptionID = qo.OptionID
JOIN OptionComponents oc ON o.OptionID = oc.OptionID
JOIN OptionComponentLang ocl ON ocl.OptionComponentID = oc.OptionComponentID
WHERE sq.SurveyID = {surveyID} AND ql.LangID = {langID} AND ocl.LangID = {langID}
                   ''')
        questions = cursor.fetchall()
        
        if questions:
            df_questions = pd.DataFrame.from_records(questions, columns=[column[0] for column in cursor.description])
            df_questions_options = df_questions.groupby(['Survey', 'SurveyID', 'QuestionID', 'Question', 'OptionType', 'OptionID']).agg({
                'OptionComponentID': lambda x: x.dropna().astype(str).tolist(),  
                'Options': lambda x: x.dropna().astype(str).tolist() 
            })
            all_questions_options.append(df_questions_options)
    
    return all_questions_options    

def generate_hw11_answers(start_date, end_date, num_answers, users, sponsorID, departmentID):
    conn = get_db_connection()
    cursor = conn.cursor()

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    generator = DataGenerator()
    description_file = './HW11/metadata.json'
    generator.generate_dataset_in_independent_mode(num_answers, description_file)
    synthetic_data = pd.DataFrame(generator.synthetic_dataset)

    sampled_users = users.sample(n=num_answers, replace=True)

    synthetic_data['UserID'] = sampled_users['UserID'].tolist()
    synthetic_data['UserProfileID'] = sampled_users['UserProfileID'].tolist()
    synthetic_data['SponsorID'] = sponsorID
    synthetic_data['DepartmentID'] = departmentID
    synthetic_data['Date'] = [random_date(start_date, end_date) for _ in range(num_answers)]

    conn.close()
    
    return synthetic_data

def insert_to_database(synthetic_data, df_user_surveys, df_question_options):
    conn = get_db_connection()
    cursor = conn.cursor()
    for index, row in synthetic_data.iterrows():
        df_answers = pd.DataFrame()
        df_sponsor_invite = pd.DataFrame()
        
        projectRoundUserID = df_user_surveys[df_user_surveys['UserID'] == row['UserID']]['ProjectRoundUserID'].values[0]
        UserID = synthetic_data["UserID"].values[index]
        UserProfileID = df_user_surveys[df_user_surveys['UserID'] == row['UserID']]['UserProfileID'].values[0]
        ProjectRoundID = df_user_surveys[df_user_surveys['ProjectRoundUserID'] == projectRoundUserID]['ProjectRoundID'].values[0]
        ProjectRoundUnitID = df_user_surveys[df_user_surveys['ProjectRoundUserID'] == projectRoundUserID]['ProjectRoundUnitID'].values[0] 
        cursor.execute("USE eform")
        cursor.execute(f'''
            SELECT TOP 1 AnswerID,
                EndDT,
                REPLACE(CONVERT(VARCHAR(255),AnswerKey),'-','') AS AnswerKey
            FROM Answer
            WHERE ProjectRoundUserID ={projectRoundUserID}
            ORDER BY AnswerID DESC                   
                    ''')
        results = cursor.fetchall()
        df_answers = pd.DataFrame.from_records(results, columns=[column[0] for column in cursor.description])
        print("df_answers", df_answers)
        
        if df_answers.empty or df_answers["EndDT"].values[0]:
            print("No previous answer found, creating new answer")
            cursor.execute("USE eform")
            cursor.execute(f'''
                INSERT INTO Answer (ProjectRoundID, ProjectRoundUnitID, ProjectRoundUserID, ExtendedFirst)
                VALUES ({ProjectRoundID}, {ProjectRoundUnitID}, {projectRoundUserID}, NULL)         
            ''')
            conn.commit()
            
            cursor.execute("USE eform")
            cursor.execute(f'''
                SELECT TOP 1 AnswerID,
                    REPLACE(CONVERT(VARCHAR(255),AnswerKey),'-','') AS AnswerKey,
                    EndDT
                FROM Answer
                WHERE ProjectRoundUserID ={projectRoundUserID} 
                ORDER BY AnswerID DESC           
                        ''')
            results = cursor.fetchall()
            df_answers = pd.DataFrame.from_records(results, columns=[column[0] for column in cursor.description])
            print("df_answers", df_answers)
            
        AnswerID = df_answers["AnswerID"].values[0]
        print("AnswerID", AnswerID)
        AnswerKey = df_answers["AnswerKey"].values[0]  
        
        # save answers to answervalue table
        for column in synthetic_data.columns:
            df_previous_answer = pd.DataFrame()
            
            if column not in ["UserID", "UserProfileID", "SponsorID", "DepartmentID", "Date"]:
                QuestionID = df_question_options[df_question_options['QuestionID'] == int(column)]['QuestionID'].values[0]
                OptionID = df_question_options[df_question_options['QuestionID'] == int(column)]['OptionID'].values[0]
                OptionType = df_question_options[df_question_options['QuestionID'] == int(column)]['OptionType'].values[0]
                Value = synthetic_data[column].values[index]
                CreatedSessionID = 0
                # check if there is a previous answer for this question
                cursor.execute("USE eform")
                cursor.execute(f'''
                    SELECT ValueInt,
                        ValueText,
                        ValueDecimal,
                        AnswerValue
                    FROM AnswerValue
                    WHERE AnswerID = {AnswerID}
                    AND QuestionID = {QuestionID}
                    AND OptionID = {OptionID}
                    AND DeletedSessionID IS NULL          
                ''')
                results = cursor.fetchall()
                df_previous_answer = pd.DataFrame.from_records(results, columns=[column[0] for column in cursor.description])
                if not df_previous_answer.empty:
                    # if there is a previous answer, check if the value is different
                    match OptionType:
                        case 1 | 3 | 9: #single choice multi choice VAS
                            cursor.execute("USE eform")
                            cursor.execute(f'''
                                UPDATE AnswerValue SET DeletedSessionID = {CreatedSessionID}
                                WHERE AnswerID = {AnswerID}
                                AND QuestionID = {QuestionID}
                                AND OptionID = {OptionID}
                                AND DeletedSessionID IS NULL
                                        ''')
                            conn.commit()
                            
                            cursor.execute(f'''
                                UPDATE AnswerValue SET ValueInt = {Value}
                                WHERE AnswerID = {AnswerID}
                                AND QuestionID = {QuestionID}
                                AND OptionID = {OptionID}
                                AND DeletedSessionID IS NULL
                                        ''')
                            conn.commit()
                            
                            print("Previous answer AnsweValue id", df_previous_answer['AnswerValue'].values[0], "from ", df_previous_answer['ValueInt'].values[0], "to", Value)
                        case 2: # free text
                            print("Save as SaveValueText")
                        case 4: # numeric
                            print("Save as SaveValueDecimal")
                        case _:
                            print("Unknown OptionType")
                else:
                    match OptionType:
                        case 1 | 3 | 9: #single choice multi choice VAS
                            cursor.execute("USE eform")
                            cursor.execute(f'''
                                INSERT INTO AnswerValue (AnswerID, QuestionID, OptionID, ValueInt, CreatedSessionID, LID)
                                VALUES ({AnswerID}, {QuestionID}, {OptionID}, {Value}, {CreatedSessionID}, {langID})
                            ''')   
                            conn.commit()
                            print("Save as SaveValueInt")
                        case 2: # free text
                            print("Save as SaveValueText")
                        case 4: # numeric
                            print("Save as SaveValueDecimal")
                        case _:
                            print("Unknown OptionType")
                    
            
            else:
                print(f"Warning: Column '{column}' not found in question options data")
            
        endDate = synthetic_data["Date"].values[index]
        dt_obj = datetime.strptime(endDate, '%Y-%m-%d %H:%M:%S.%f') 
        cursor.execute(f"UPDATE Answer SET EndDT = ? WHERE AnswerID = {AnswerID}", (dt_obj,))
        # cursor.execute("USE eform")
        # cursor.execute(f'''
        #     UPDATE Answer SET EndDT = GETDATE()
        #     WHERE AnswerID = {AnswerID}
        #                 ''')
        # conn.commit()
        print("End date updated to", endDate)
            
        # check if active
        cursor.execute("USE healthwatch")
        cursor.execute(f'''
            SELECT SponsorInviteID,
                StoppedReason,
                StoppedPercent
            FROM SponsorInvite
            WHERE SponsorID = {sponsorID} AND UserID = {UserID}
            ''')
        results = cursor.fetchall()
        df_sponsor_invite = pd.DataFrame.from_records(results, columns=[column[0] for column in cursor.description])
        isInactive = df_sponsor_invite['StoppedReason'].values[0] if not df_sponsor_invite.empty else None
        location = random.randint(1, 2)
            # need pa ang locationValue here
        cursor.execute("USE healthwatch")
        cursor.execute(f'''
            INSERT INTO UserProjectRoundUserAnswer (ProjectRoundUserID, AnswerKey, UserProfileID, AnswerID, Inactive, Location)
            VALUES ({projectRoundUserID}, '{AnswerKey}', {UserProfileID}, {AnswerID}, 0, NULL)
                        ''')
        conn.commit()

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
st.title("BulkUser - Self Answering Robot")

langOptions = ["1 - Swedish", "2 - English"]
selected_lang = st.selectbox("Select Language", langOptions)
langID = int(selected_lang.split(" - ")[0])

sponsor_dict = get_sponsor_data()
sponsor_options = [f"{name} - ({info['NumUsers']} users)" for name, info in sponsor_dict.items()]

sponsor_options = ["Select a sponsor..."] + sponsor_options
selected_option = st.selectbox("Select Sponsor", sponsor_options)

if selected_option != "Select a sponsor...":
    # selected_sponsor_name = selected_option.split(" - ")[0]
    selected_sponsor_name = re.match(r"(.*) - \(\d+ users\)", selected_option).group(1)
    sponsor_info = sponsor_dict[selected_sponsor_name]
    sponsorID = sponsor_info["SponsorID"]

    department_dict = get_department_ids(sponsorID)

    if not department_dict:
        st.warning(f"No departments available for {selected_sponsor_name}.")
    else:
        department_options = [f"{name} - {info['NumUsers']}" for name, info in department_dict.items()]
        
        selected_department_option = st.selectbox("Select Department", department_options)

        selected_department_name = selected_department_option.split(" - ")[0]

        department_info = department_dict[selected_department_name]
        departmentID = department_info["DepartmentID"]
        
        users = get_users_info(sponsorID, departmentID, langID)
        
        if users.empty:
            st.warning(f"No users available for {selected_department_name}.")
        else:
            st.dataframe(users, use_container_width=True)
            
            survey_ids = users['SurveyID'].unique()
            survey_questions_and_options = get_survey_questions_and_options(survey_ids, langID)
                        
            survey_names = []
            for i, df in enumerate(survey_questions_and_options):
                if not df.empty and 'Survey' in df.columns:
                    survey_name = df['Survey'].unique()[0]
                    survey_id = df['SurveyID'].unique()[0]
                    survey_names.append(f"{survey_name} (ID: {survey_id})")
                else:
                    survey_names.append(f"Survey {i+1}")
                    
            selected_survey_index = st.selectbox(
                "Select Survey to View", 
                range(len(survey_names)),
                format_func=lambda i: survey_names[i]
            )
            
            selected_df = survey_questions_and_options[selected_survey_index]
            st.write(f"Viewing questions for {survey_names[selected_survey_index]}")
            st.dataframe(selected_df, use_container_width=True)
            
            selected_df_reset = selected_df.reset_index()
            
            users = users[users['SurveyID'] == selected_df_reset['SurveyID'].values[0]]
            users = users[users['ProjectRoundUserID'].notnull()]
            
            if len(selected_df) == 11 and 'OptionType' in selected_df_reset.columns and (selected_df_reset['OptionType'] == 9).all():
                num_answers = st.number_input("Number of Answers", min_value=1, value=1)
                
                start_date = str(st.date_input("Start Date", max_value=pd.Timestamp.today().date()))
                end_date = str(st.date_input("End Date", min_value=start_date, max_value=pd.Timestamp.today().date()))
                
                if st.button("Generate Data"):
                    synthetic_data = generate_hw11_answers(start_date, end_date, num_answers, users, sponsorID, departmentID)
                    st.session_state.synthetic_data = synthetic_data
                    st.dataframe(synthetic_data)
                    csv_data = synthetic_data.to_csv(index=False)
                    st.download_button("Download Data as CSV", csv_data, "synthetic_data.csv", "text/csv")
                    
                if 'synthetic_data' in st.session_state and st.button("Insert to Database"):
                    timer_placeholder = st.empty()
                    start_time = time.time()
                    
                    st.session_state['timer_running'] = True
                    
                    timer_placeholder.info("Starting database insertion...")
                    
                    def update_timer():
                        while 'timer_running' in st.session_state and st.session_state['timer_running']:
                            elapsed = time.time() - start_time
                            mins, secs = divmod(elapsed, 60)
                            timer_placeholder.info(f"⏱️ Time elapsed: {int(mins)}m {int(secs)}s")
                            time.sleep(0.5)
                            
                    timer_thread = threading.Thread(target=update_timer, daemon=True)
                    timer_thread.start()
                    
                    try:
                        insert_to_database(st.session_state.synthetic_data, users, selected_df_reset)
                        st.session_state['timer_running'] = False
                        elapsed_total = time.time() - start_time
                        minutes, seconds = divmod(elapsed_total, 60)
                        timer_placeholder.success(f"✅ Database insertion completed in {int(minutes)}m {int(seconds)}s")
                        if 'synthetic_data' in st.session_state:
                            del st.session_state['synthetic_data']
                    except Exception as e:
                        st.session_state['timer_running'] = False
                        timer_placeholder.error(f"❌ Error during database insertion: {str(e)}")
    