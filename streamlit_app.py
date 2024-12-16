import streamlit as st  # Import the Streamlit library

# Set the title of the app
st.title("Simple Streamlit App")

# Create a sidebar for user input
st.sidebar.header("User Input")
user_name = st.sidebar.text_input("Enter your name:", "Guest")

# Create a slider in the sidebar to get a number input
user_number = st.sidebar.slider("Select a number:", 0, 100, 50)

# Display a welcome message in the main area of the app
st.write(f"Hello, {user_name}! Welcome to the app.")

# Display the square of the selected number
st.write(f"The square of {user_number} is {user_number ** 2}.")

# Add an expander for additional information
with st.expander("About this app"):
    st.write("This is a simple Streamlit app with user input and interactivity.")

# Add a button and display a message when it's clicked
if st.button("Click me"):
    st.write("You clicked the button!")

# Run this file with `streamlit run filename.py` to test locally.