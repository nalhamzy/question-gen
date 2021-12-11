

from pandas.core.frame import DataFrame
from serpapi import GoogleSearch
import os
import csv
import time
import pandas as pd

import streamlit as st
import numpy as np
from bs4 import BeautifulSoup
from requests import get
import re 
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from pprint import pprint
from requests import session
from Questgen import main as qgen
app_secret = 'e478f284e8aa736bc21fd8691ae7d08f14680d2e6a1fac7a8d6ad1f51e1b358f'

#nltk.download('stopwords')
def load_nltk():
     if 'qg' not in st.session_state:
        st.session_state['qg'] = qgen.QGen()
        return st.session_state['qg']
     else:
         return st.session_state['qg']

  
no_paragraphs = st.slider('No. of paragraphs:',10, 500,10)

def is_valid_sentence(input_text):
    s  = re.search('(([A-Za-z]){2,}\s+.*){5}',input_text)
    return s != None


def convert_df(df):
    return df.to_csv().encode('utf-8')
user_input = st.text_input('Google Search', 'Coffee Table')
def search_keywords(input_text):
    gsearch = GoogleSearch({
        "q": input_text, 
        "location": "Austin,Texas",
        "num" : "15",
        "api_key": app_secret
    })
    result = gsearch.get_dict()
    #print(result)
    final_results= []

    df = pd.DataFrame(columns=['link','title','text'])
    
    n_samples = 5
    print('len')
    print(len(result['organic_results']))
    for item in result['organic_results']:
        print(item['link'])
        page_url = item['link']
        title=item['title']
        response = get(page_url)
        html_soup = BeautifulSoup(response.text, 'html.parser')
        p_list = html_soup.find_all('p')[:no_paragraphs]
        st.write("\n")
        st.write("Website title: %s :"%title) 
        text = ''
        for p in p_list:

            st.write('<p>' + p.text + '<\p>')
            if is_valid_sentence(p.text):
                text += p.text + ' '
        final_results.append(text)
        if len(p_list) > 1 and text != '':
            df.loc[len(df)] = {'link':page_url,'title':title,'text':text}
        if n_samples <= 0: 
            break
        n_samples -= 1
        
    st.write(df)
    return df




def main():
    qg = load_nltk()
    df_qes = pd.DataFrame(columns=['link','title','question','answer','context'])
    if st.button("Search"):
        if user_input != None:
            results = search_keywords(user_input)
            csv = convert_df(results)

            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='extracted_text.csv',
                mime='text/csv',
            ) 

            
            for idx, row in results.iterrows():
                payload = {
                    "input_text": row['text']
                }
                output = qg.predict_shortq(payload)
                st.write(output)
                if "questions" in output:
                    for item in output['questions']:
                        df_qes.loc[len(df_qes)] = {'link':row['link'],'title':row['title'],"question":item['Question'],'answer':item['Answer'],'context':item['context']}   
            st.write(df_qes)

            if len(df_qes) > 0:
                csv = convert_df(df_qes)

                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name='qa_pairs.csv',
                    mime='text/csv',
                ) 
if __name__ == '__main__':
	main()
 