import os
from dotenv import load_dotenv

import tiktoken
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
# from openai.embeddings_utils import cosine_similarity

# Use the environment variables to retrieve API keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings()

import pandas as pd
df = pd.read_csv("Data.csv")
print(df)

df['embedding'] = df['Words'].apply(lambda x: embeddings.embed_query(x))
df.to_csv('word_embeddings.csv')

new_df = pd.read_csv("word_embeddings.csv")
print(new_df)

our_text = "Soccer"
text_embedding = embeddings.embed_query(our_text)
print(f"Our embedding is {text_embedding}")

from openai.embeddings_utils import cosine_similarity
df["similarity score"] = df["embedding"].apply(lambda x: cosine_similarity(x, text_embedding))
df.sort_values("similarity score", ascending=False).head()