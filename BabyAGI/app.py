import os
from collections import deque
from typing import Dict, List, Optional, Any

from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from langchain.chains.base import Chain
from langchain.experimental import BabyAGI

from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore

# Define your embedding model
embeddings_model = OpenAIEmbeddings()
# Initialize the vectorstore as empty
import faiss

embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

with open('/Users/ravikala/BabyAGI/financial_transaction.txt', 'r') as file:
    dataset = file.read()

OBJECTIVE = "Can I afford a home of 1 Crore and how much installment can I afford? Use one month financial_transaction data here {dataset}\
			"

llm = OpenAI(temperature=0)

# Logging of LLMChains
verbose = False
# If None, will keep on going forever
max_iterations: Optional[int] = 5
baby_agi = BabyAGI.from_llm(
    llm=llm, vectorstore=vectorstore, verbose=verbose, max_iterations=max_iterations
)

baby_agi({"objective": OBJECTIVE})

print()