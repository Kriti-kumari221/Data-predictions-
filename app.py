import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

st.title("🏢 Building Data Preprocessing")
st.write("📂 Upload your housing dataset (CSV) to begin preprocessing")

upload_file = st.file_uploader("Upload CSV file", type='csv')#use for the CSVfile to upload 

if upload_file:
    data = pd.read_csv(upload_file)#it will read the csv file 
    st.subheader("📄 Original Data")
    st.write(data.head())#it will show the data from the header without preprocessing 

    # Drop 'society' column if it exists and we will remove unnessessery column from the data so that it can get efficient 
    if "society" in data.columns:
        data.drop("society", axis=1, inplace=True)

    # It will Fill the missing values using forward fill
    data.fillna(method="ffill", inplace=True)

    # Extract number from 'size' column like "2 BHK", "4 Bedroom because machine learning is working only for the numeric "
    data["size"] = data["size"].str.extract('(\d+)').astype(float)

    # Convert total_sqft to numeric there is "-" so it will generate error so we have to convert it in fully numeric data
    def convert_sqft_to_num(x):
        try:
            x = str(x)#convert to string for any number for example 453="458", 89-100="89-100"
            if '-' in x:#check for "-" if get in string ie x
                tokens = x.split('-')#then it will split like 89-100 to ["89","100"]
                return (float(tokens[0]) + float(tokens[1])) / 2#and return it using avg becouse we don't no the exact value
            else:
                return float(x)#if x is not string then convert it in float and return 
        except:
            return None# the value like this 567hu 45sd then return none so it will not throw an error

    data['total_sqft'] = data['total_sqft'].apply(convert_sqft_to_num) #apply is use here for the every column 
    data['total_sqft'].fillna(data['total_sqft'].median(), inplace=True) #now it will fill the empty data 
    #using median becouse the data is continuously increasing  and not like a random so we cant use forward fill means ffill

    # One-hot encode categorical variables now it will convert catogorical data to the numerical like 0 or 1 o for not present 1 for present 
    categorical_cols = ["area_type", "availability", "location"]
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)#drop first is use for the if first and second will get the 1 or 0 the third one will automaticaly disided 

    # Scale numeric features
    scaler = StandardScaler()#use for the scale
    numeric_cols = ['total_sqft', 'bath', 'balcony', 'size']
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])#fit is use for the mean and standerd deviation   
    # transform use for the scal z=(x-mean)//standerd deviation 
    st.subheader("✅ Cleaned & Preprocessed Data")
    st.write(data.head())

