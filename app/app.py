import sqlite3

import streamlit as st

from search import SearchSystem
from recommendations import RecommenderSystem


def get_item(item_id):
    with sqlite3.connect("../data/items.db") as conn:
        c = conn.cursor()
        c.row_factory = sqlite3.Row
        c.execute(f"select * from items where item_id like '{item_id}'")
        return c.fetchone()


def show_item(item_id):
    item = get_item(item_id)
    title = item["title"]
    with st.container(border=True):
        st.write(f"**{title}**")
        st.write(item["description"])
        if st.button("Recommend similar items", key=item["item_id"]):
            st.session_state["recommendation_query"] = item["item_id"]
            st.session_state["search_query"] = None  # reset
            st.rerun()


def main():
    st.title("Graph-based RecSys")

    if "searchsys" not in st.session_state:
        st.session_state["searchsys"] = SearchSystem(
            items_path="../data/items.db")
    if "recsys" not in st.session_state:
        st.session_state["recsys"] = RecommenderSystem(
            faiss_index_path="../data/index.faiss",
            embeddings_path="../data/embeddings.npy")

    if "search_query" not in st.session_state:
        st.session_state["search_query"] = None
    if "recommendation_query" not in st.session_state:
        st.session_state["recommendation_query"] = None

    search_query = st.text_input("Enter item name", st.session_state["search_query"])

    if st.button("Search"):
        st.session_state["search_query"] = search_query
        st.session_state["recommendation_query"] = None  # reset

    if st.session_state["recommendation_query"] is not None:
        recommendation_query = st.session_state["recommendation_query"]
        recomendation_query_title = get_item(recommendation_query)["title"]
        st.subheader(f'Recommendation Results for "{recomendation_query_title}"')
        results = st.session_state["recsys"].recommend_items(recommendation_query)
        for item_id in results:
            show_item(item_id)

    elif st.session_state["search_query"] is not None:
        search_query = st.session_state["search_query"]
        st.subheader(f'Search Results for "{search_query}"')
        results = st.session_state["searchsys"].search_items(search_query)
        for item_id in results:
            show_item(item_id)

if __name__ == "__main__":
    main()