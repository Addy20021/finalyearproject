import streamlit as st
from setup_db import get_user

def login():
    st.header("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = get_user(username)
        if user and user[1] == password:
            st.session_state.logged_in = True
            st.success(f"Welcome {username}!")
        else:
            st.error("Invalid username or password.")
