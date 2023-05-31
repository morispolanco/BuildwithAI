from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import pandas as pd
from langchain.agents import create_csv_agent
import io
import csv
import time
import matplotlib.pyplot as plt

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your data")
    st.header("EDAGPT: Explanatory data analysis using ChatGPT 💬")
    
    # upload file
    

    csv_file = st.file_uploader("Upload your data", type="csv")

    if csv_file is not None:
      csv_buffer = io.StringIO(csv_file.read().decode('utf-8'))
      csv_df = pd.read_csv(csv_buffer)
    

      csv_df.to_csv('/users/ravikala/EDAGPT/df.csv', index=False)
      
      time.sleep(3)

      agent = create_csv_agent(OpenAI(temperature=0), '/users/ravikala/EDAGPT/df.csv', verbose=True)
      
      user_question = st.text_input("Ask a question about your data:")

      output=agent.run(user_question)

      time.sleep(5)


      plt.savefig('/users/ravikala/EDAGPT/plot.png')
      
      time.sleep(2)

      st.image('/users/ravikala/EDAGPT/plot.png')

      time.sleep(2)

      st.write(output)

if __name__ == '__main__':
    main()