import streamlit as st
import pandas as pd
import numpy as np
import faiss


items_data = pd.read_csv("../data/items.csv")


@st.cache_data
def get_embeddings():
    return np.load("../data/embeddings.npy")


@st.cache_resource()
def get_faiss_index():
    index = faiss.read_index("../data/index.faiss")
    return index