import streamlit as st

from data import items_data
from search import search_items
from recommendation import recommend_items


def show_item(item_id):
    item = items_data.iloc[item_id, :]
    title = item["title"]
    st.write(f"**{title}**")
    if st.button("Recommend similar items", key=item["item_id"]):
        st.session_state["recommendation_query"] = item["item_id"]
        st.session_state["search_query"] = None  # reset
        st.rerun()
    st.write("---") 


def main():
    st.title("Graph-based Recommendation System")
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
        st.subheader(f'Recommendation Results for "{recommendation_query}"')
        results = recommend_items(recommendation_query)
        for item_id in results:
            show_item(item_id)

    elif st.session_state["search_query"] is not None:
        search_query = st.session_state["search_query"]
        st.subheader(f'Search Results for "{search_query}"')
        results = search_items(search_query)
        for item_id in results:
            show_item(item_id)

if __name__ == "__main__":
    main()