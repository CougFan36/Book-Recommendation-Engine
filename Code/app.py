import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
import random

# Pulling in merged_df from data via pickling
with open('../Code/merged_dataframe.pkl', 'rb') as m:
    merged_df = pickle.load(m)

# Pull in item_encoder via pickling
with open('../Code/item_encoder.pkl', 'rb') as f:
    item_encoder = pickle.load(f)

# Pull in user_encoder via pickling
with open('../Code/user_encoder.pkl', 'rb') as u:
    user_encoder = pickle.load(u)

# Pull in train_matrix via pickling
with open('../Code/train_matrix.pkl', 'rb') as t:
    train_matrix = pickle.load(t)

ncf_model = load_model('../Data/recc_model.h5')

train_matrix = train_matrix.merge(merged_df[['Id', 'Title']], left_on='item_id_unenc', right_on='Id', how='left')
# To enable autocompletion, we'll use a library called streamlit that now includes elements for
# enhancing text input with suggestions.


all_items = train_matrix['item_id'].unique()
all_users = train_matrix['user_id'].unique()

def get_top_n_recommendations(model, user_id, item_id, all_items, top_n=10):
    # Generate an array where the user_id and item_id are repeated for each item
    user_array = np.array([user_id] * len(all_items))
    items_array = np.array(all_items)   

    # Predict the interaction scores for these user-item pairs
    predictions = model.predict([user_array, items_array], batch_size=128, verbose=1)
   
    # Flatten the predictions to get a 1D array
    predictions = predictions.flatten()

    # Get top `n` items with highest prediction scores, except the input item_id
    sorted_indices = predictions.argsort()[::-1]
    top_n_items = [all_items[i] for i in sorted_indices if all_items[i] != item_id][:top_n]
    return top_n_items

# Function to get book titles from book_ids
#def get_titles_from_ids(id_arr):
#    original_book_ids = item_encoder.inverse_transform(id_arr)
#    book_titles = merged_df[merged_df['Id'].isin(original_book_ids)]['Title'].unique()
#    return book_titles
def get_titles_from_ids(id_arr):
    original_book_ids = train_matrix.loc[train_matrix['item_id'].isin(id_arr), 'Id']
    book_titles = train_matrix[train_matrix['Id'].isin(original_book_ids)]['Title'].unique()
    return book_titles

# Get the book titles based on the list of book ids
#books_col = merged_df['Title'].astype(str)
titles = get_titles_from_ids(train_matrix['item_id'].unique())
st.title("Hobbit Recommender")

# Text input for movie title
selected_book = st.text_input("Enter a book title:", "")

if selected_book != "":
    # Autocomplete functionality using selectbox
    matching_books = [book for book in titles if selected_book.lower() in book.lower()]
    if matching_books:
        selected_book = st.selectbox("Select a book from suggestions:", matching_books)

    # Get recommendations
    if st.button("Get Recommendations"):
    # Map book title to encoded ID and run recommender
        book_encoded = train_matrix.loc[train_matrix['Title']==selected_book, 'item_id'].values[0]
        selected_users = train_matrix.loc[train_matrix['item_id']==book_encoded, 'user_id'].values[0]
        recommendations = get_top_n_recommendations(ncf_model, selected_users, book_encoded, all_items)

        if recommendations:
        # Grab recommendation list and convert back to book IDs and then back to titles, then print titles
            st.write("We recommend you read these books:")
            recommendation_titles = get_titles_from_ids(recommendations)   
            for book in recommendation_titles:

                st.write(book)
        else:
            st.write("No recommendations found. Please check the book title.")
