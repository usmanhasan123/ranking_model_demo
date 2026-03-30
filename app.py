import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import tempfile

import gspread
from oauth2client.service_account import ServiceAccountCredentials
from google.oauth2.service_account import Credentials

# Load model
model = joblib.load("model/model.pkl")

# Dummy item list (replace later with real items)
ITEMS = list(range(1, 51))

st.title("Recommender System Demo")

user_id = st.number_input("Enter User ID", min_value=1, max_value=100, value=1)

# def generate_features(user_id, items):
#     df = pd.DataFrame({
#         "user_id": [user_id]*len(items),
#         "item_id": items,
#     })

#     # simple placeholders (same logic as training)
#     df["item_popularity"] = np.random.randint(1, 100, len(items))
#     df["user_activity"] = np.random.randint(1, 50, len(items))

#     return df

def connect_to_gsheet(creds_json,spreadsheet_name):
    scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
             "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
    
    credentials = ServiceAccountCredentials.from_json_keyfile_name(creds_json, scope)
    client = gspread.authorize(credentials)
    spreadsheet = client.open(spreadsheet_name)  # Access the first sheet
    return spreadsheet

if st.button("Get Recommendations"):
    # features = generate_features(user_id, ITEMS)
    
    # private_key_json=os.getenv('private_key_json')
    # private_key_json=st.secrets['private_key_json']
    # st.write(private_key_json)

    os.environ["private_key_json"] = st.secrets["private_key_json"]
    private_key_json = os.getenv("private_key_json")
    st.write(private_key_json)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp.write(private_key_json.encode())  # write bytes
        tmp.flush()
        creds_path = tmp.name
        
    SPREADSHEET_NAME = 'Offline Data'
    # SPREADSHEET_NAME_2 = 'Tab 2 Data'
    # SHEET_NAME = 'Sheet1'
    CREDENTIALS_FILE = creds_path #'./private_key.json'
    sheet_by_name = connect_to_gsheet(CREDENTIALS_FILE, SPREADSHEET_NAME)
    
    ws = sheet_by_name.worksheet("user_data")
    x=ws.get_all_records()
    user_data=pd.DataFrame(x)

    ws = sheet_by_name.worksheet("item_data")
    x=ws.get_all_records()
    item_data=pd.DataFrame(x)
    # st.write(user_data.head())
    features=user_data[user_data['user_id']==user_id]
    features=features=features.merge(item_data, how='cross')

    scores = model.predict_proba(
        features[["user_id", "item_id", "user_total_interactions","user_unique_items","user_unique_categories","user_avg_price","user_avg_rating","user_price_std",
       "user_rating_std","item_total_interactions","item_unique_users","item_avg_price","item_avg_rating","item_price_std","item_rating_std",
       "item_price","item_rating"]]
    )[:, 1]

    features["score"] = scores
    features["EV"]=scores*features["item_price"]

    top_items = features.sort_values("EV", ascending=False).head(5)

    st.write("### Top Recommendations")
    st.write(top_items[["item_id", "score", "EV"]])
