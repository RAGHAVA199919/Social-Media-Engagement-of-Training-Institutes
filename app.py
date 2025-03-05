import streamlit as st
import pandas as pd
import joblib

# Load the trained KNN model
KNN = joblib.load('knn_model.pkl')

# Streamlit UI
def main():
    st.set_page_config(page_title="Engagement Rate Predictor", page_icon="📊", layout="centered")
    
    # Add a banner image
    st.image("https://i0.wp.com/slajobs.com/wp-content/uploads/2019/05/internship-training-in-chennai-kk-nagar.jpg?w=625&ssl=17", use_column_width=True)
    
    st.title("📈 Engagement Rate Prediction App for Training Institutes")
    st.write("### Enter the details below to predict the engagement rate.")
    
    # Sidebar with an image
    st.sidebar.image("https://www.webskittersacademy.in/wp-content/uploads/2022/04/IT-Training-Institute.jpg", use_column_width=True)
    st.sidebar.header("🔍 About the Model")
    st.sidebar.write("This model uses K-Nearest Neighbors (KNN) regression to predict engagement rates based on various input features.")
    
    # User inputs
    course_name = st.text_input("📚 Course Name:")
    course_type = st.text_input("📌 Course Type:")
    followers = st.number_input("👥 Number of Followers:", min_value=0)
    likes = st.number_input("👍 Number of Likes:", min_value=0)
    comments = st.number_input("💬 Number of Comments:", min_value=0)
    shares = st.number_input("🔄 Number of Shares:", min_value=0)
    total_engagement = st.number_input("📊 Total Engagement:", min_value=0)
    
    # Predict button
    if st.button("🚀 Predict Engagement Rate"):
        input_data = pd.DataFrame([[course_name, course_type, followers, likes, comments, shares, total_engagement]],
                                  columns=['CourseName', 'type', 'Followers', 'Likes', 'Comments', 'Share', 'Total Engagement'])
        
        predicted_engagement_rate = KNN.predict(input_data)
        
        st.success(f"🎯 Predicted Engagement Rate: {predicted_engagement_rate[0]:.4f}")
        

if __name__ == "__main__":
    main()
