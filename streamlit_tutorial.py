import streamlit as st 
import pandas as pd 
import numpy as np

st.title("Hello, Streamlit!")
st.write(":streamlit: This is your first streamlit app")
st.text("Lets get started")


name = st.text_input("Enter your name : ")
if st.button("Greet"):
    st.success(f"Hello, {name}")


#Displaying Data and Charts 
df = pd.DataFrame(np.random.randn(10,2), columns=['A','B'])
st.line_chart(df)
st.bar_chart(df)

#File uploading and caching topics 
upload_file = st.file_uploader("upload a csv", type='csv')
if upload_file:
    df = pd.read_csv(upload_file)
    st.dataframe(df)

st.title("Text and Markdown Demo")
st.header("This is a header")
st.subheader("This is a subheader")
st.markdown("**Bold**, *Italic*, `Code`, [Link](https://streamlit.io)")
st.code("for i in range(5): print(i)", language='python')

st.text_input("What's your name?")
st.text_area("Write something...")
st.number_input("Pick a number", min_value=0, max_value=100)
st.slider("Choose a range", 0, 100)
st.selectbox("Select a fruit", ["Apple", "Banana", "Cherry", "Mango", "Elderberry"])
st.multiselect("Choose toppings", ["Cheese", "Tomato", "Olives", "Mushrooms", "Jalapenos"])
st.radio("Pick one", ["Option A", "Option B", "Option C"])
st.checkbox("I agree to the terms and conditions")

if st.checkbox("Show Details"):
    st.info("Here are more details...")

option = st.radio("Choose view", ["Show Chart", "Show Table"])
if option == "Show Chart":
    st.write("Chart would appear here.")
else:
    st.write("Table would appear here.")

with st.form("Login_Form"):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    submitted = st.form_submit_button("Login")

    if submitted:
        st.success(f"Welcome, {username}!")

st.image("https://shorturl.at/uSrvY", caption="Sample Image", use_column_width=True)
st.video("https://www.youtube.com/watch?v=8RSNyIWZ6MQ&pp=ugUEEgJoaQ%3D%3D")