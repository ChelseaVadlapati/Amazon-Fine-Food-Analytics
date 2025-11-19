import streamlit as st

st.title("Amazon Fine Food Sentiment – smoke test")
st.write("If you can see this on Streamlit Cloud, the environment is fine.")

review = st.text_area("Type anything here")
if st.button("Echo"):
    st.write("You typed:", review)
