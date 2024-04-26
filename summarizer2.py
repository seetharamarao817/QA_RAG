#import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN,AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from langchain_community.document_loaders.parsers.pdf import  PDFPlumberParser

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langchain.text_splitter import RecursiveCharacterTextSplitter
import nltk
from nltk.tokenize import sent_tokenize

import sqlite3

from langchain_community.chat_models.openai import ChatOpenAI
from langchain import hub
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os
OPEN_API_KEY = ''
os.environ["OPENAI_API_KEY"]  = OPEN_API_KEY
import nltk
from nltk.tokenize import sent_tokenize

def generate_summary(text):
    max_tokens = 16384  # Adjust based on your model's token limit
    # Split text into sentences
    sentences = sent_tokenize(text)
    current_chunk = ""
    summaries = []

    for sentence in sentences:
        # Check if adding this sentence would exceed the max token limit
        if len((current_chunk + " " + sentence).split()) > max_tokens:
            # Summarize the current chunk and reset it
            summaries.append(summarize_chunk(current_chunk))
            current_chunk = sentence
        else:
            current_chunk += " " + sentence

    # Summarize the last chunk if it's not empty
    if current_chunk:
        summaries.append(summarize_chunk(current_chunk))

    # Combine all summaries into a single summary
    final_summary = " ".join(summaries)
    return final_summary

def summarize_chunk(chunk):
    map_template = """The following is a set of document:
{docs}
Based on these documents and the please write a concise summary.
The summary should be accurate, relevant, and concise. It should avoid redundancy and also faithfully represent the key points from the documents.
Write a detailed and concise summary of the provided text, ensuring all critical information is included.
Helpful Answer:"""
    map_prompt = PromptTemplate(template=map_template,input_variables=['docs'])
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Chain
    rag_chain = map_prompt | llm | StrOutputParser()

    # Run
    generation = rag_chain.invoke({'docs':chunk})

    return generation


# Initialize NLP tools
nltk.download('punkt')
nltk.download('stopwords')


def text_split(text, max_words=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i+max_words]))
    return chunks
    return docs

# Convert texts to vectors
def texts_to_vectors(texts, model):
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings

def cluster_texts(embeddings, n_clusters=2):
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
    cluster_labels = clustering_model.fit_predict(embeddings.cpu().numpy())
    return cluster_labels

# Extract keywords using TF-IDF
def extract_keywords(text):
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    freq_dist = nltk.FreqDist(filtered_words)
    keywords = [word for word, count in freq_dist.most_common(6)]
    return keywords

# Database operations
def initialize_db():
    conn = sqlite3.connect('agenda_summary.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS summaries (cluster_id INTEGER, summary TEXT, keywords TEXT)''')
    return conn

def save_summary_to_db(conn, cluster_id, summary, keywords):
    c = conn.cursor()
    c.execute('INSERT INTO summaries (cluster_id, summary, keywords) VALUES (?, ?, ?)', (cluster_id, summary, ', '.join(keywords)))
    conn.commit()

# Main function to process the document
def process_document(data):
    # Load pre-trained SBERT model for embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    page_chunks = []
    for page in data:
        # Split the page content and store the chunks
        chunks = text_split(page.page_content)
        page_chunks.extend(chunks)
    # Convert texts to TF-IDF vectors
    tfidf_matrix = texts_to_vectors(page_chunks,model)

    # Cluster embeddings
    cluster_labels = cluster_texts(tfidf_matrix)

    # Initialize database
    conn = initialize_db()

    # Process each cluster
    for cluster_id in np.unique(cluster_labels):
        # Collect texts for the current cluster
        cluster_text = [page_chunks[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
        cluster_text = " ".join(cluster_text)  # Combine texts into a single string for summarization

        summary = generate_summary(cluster_text)
        keywords = extract_keywords(cluster_text)
        save_summary_to_db(conn, cluster_id, summary, keywords)


def extract_query_keywords(query):
    # Tokenize and remove stopwords
    words = word_tokenize(query.lower())
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    return filtered_words

def retrieve_summaries(query):
    # Extract keywords from the user's query
    query_keywords = extract_query_keywords(query)

    # Connect to the SQLite database
    conn = sqlite3.connect('agenda_summary.db')
    cursor = conn.cursor()

    # Prepare to collect matches
    matched_summaries = []

    # Retrieve keywords and summaries from the database
    cursor.execute("SELECT cluster_id, summary, keywords FROM summaries")
    records = cursor.fetchall()

    # Check for keyword matches and collect corresponding summaries
    for record in records:
        db_keywords = record[2].split(', ')
        if any(keyword in db_keywords for keyword in query_keywords):
            matched_summaries.append((record[0], record[1]))  # Collecting cluster_id and summary

    # Close the connection
    conn.close()

    return matched_summaries


def generate(query):
    
    
    documents = retrieve_summaries(query)
    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    generation = rag_chain.invoke({"context": documents, "question": query})
    return {
        "keys": {"question":query, "generation": generation}
    }