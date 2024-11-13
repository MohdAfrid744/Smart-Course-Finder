import pandas as pd

# Load the CSV file
courses = pd.read_csv('free_courses.csv')

# View the structure
print(courses.head())

from sentence_transformers import SentenceTransformer

# Load a pre-trained BERT model for embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Combine course information to generate embeddings
course_descriptions = courses['topics'] + " " + courses['keywords']
course_embeddings = model.encode(course_descriptions, convert_to_tensor=True)

from sklearn.metrics.pairwise import cosine_similarity
import torch

def search_courses(query, course_embeddings, courses, model):
    # Get query embedding
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Compute cosine similarities between the query and all courses
    similarities = cosine_similarity(query_embedding.unsqueeze(0), course_embeddings)[0]
    
    # Get top 5 most similar courses
    top_5_indices = torch.topk(torch.tensor(similarities), 5).indices
    
    # Return the most relevant courses
    return courses.iloc[top_5_indices]

import streamlit as st

# Title and user input
st.title("Smart Course Search System")
query = st.text_input("Enter keywords or a description to find the most relevant courses:")

# If query exists, perform the search
if query:
    results = search_courses(query, course_embeddings, courses, model)
    
    # Display results
    for idx, row in results.iterrows():
        st.write(f"### {row['course_name']}")
        st.write(f"**Topics:** {row['topics']}")
        st.write(f"**Keywords:** {row['keywords']}")
        st.write(f"**Difficulty:** {row['difficulty']}")
        st.write("---")
