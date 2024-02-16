import streamlit as st
import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load course data
st.set_page_config(layout="wide")

container_style = """
    <style>
        .container {
            position: absolute;
            top: 10px;
            right: 10px;
            background-color: #262631;
            padding: 10px;
            border-radius: 15px;
            border: 0.5px solid gray;
            
        }
        .link-button {
            text-decoration: none;
            color: white !important;  /* Button color */
            font-weight: bold;
            # background-color: transparent !important;
            border: none !important;
            cursor: pointer;
            outline: none !important;
            
        }
        .container:hover {
            border: 1px solid white;
            border-radius: 15px;
            
        }
    </style>
"""

# Apply the CSS style
st.markdown(container_style, unsafe_allow_html=True)

# Create the container with the link button
st.markdown("""
<div class='container'>
    <a class='link-button' href='https://streamlit.io/gallery'>Log Out</a>
</div>
""", unsafe_allow_html=True)
# CSS for center-aligning the header and styling the line
page_bg_img = '''
<style>

[data-testid="stAppViewContainer"] > .main {
    background-image: url(https://i.ibb.co/rd9qV9r/imp2-1.jpg);
 
    background-position: center;
    
   
    background-attachment: local, fixed;
}

</style>
'''


st.markdown(page_bg_img, unsafe_allow_html=True)

header_style = """
    <style>
      
     
     .header {
            color: #fff;
            padding: 0;
            margin-top: 0;
            text-align: center;
        }
        .caption {
            color: #fff;
            text-align: center;
            margin-top: 0;
            padding-top: 100px:
            font-size: 60px;
        }
        .line {
           
            border-bottom: 2px dashed #f85a40 #ccc;
            margin-bottom: 20px;
            padding-bottom: 80px;
        }
        .block-container st-emotion-cache-z5fcl4 ea3mdgi2 {
            padding: 0;
        }
        
        
            
    </style>
"""
# st.set_page_config(layout="wide")

# Adding the CSS to the Streamlit app
st.markdown(header_style, unsafe_allow_html=True)


# Header with center alignment and line separator
st.markdown("<h1 class='header'>The Curator<span style='color: #f85a40;'>.</span></h1>", unsafe_allow_html=True)


st.markdown("<h1 class='caption'>Courses Recommendation Tool</h1>", unsafe_allow_html=True)
st.markdown("<div class='line'></div>", unsafe_allow_html=True)
courses = pd.read_csv(r"Coursera.csv")

def get_course_recommendations(course_name, difficulty_level, previous_courses):
    df_excluded = courses[
        (~courses['Course Name'].isin(previous_courses)) &
        (courses['Course Name'] != course_name) &
        (courses['Difficulty Level'] == difficulty_level)
    ]
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_excluded['Course Description'].fillna(''))
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_vectorizer.transform([courses[courses['Course Name'] == course_name]['Course Description'].iloc[0]]))
    similar_courses_indices = cosine_similarities.flatten().argsort()[::-1]
    similar_courses = df_excluded.iloc[similar_courses_indices].head(5)
    return similar_courses

def get_course_details(course_name):
    course_details = courses[courses['Course Name'] == course_name].to_dict(orient='records')[0]
    return course_details

# Streamlit app
def app():

    # Select a course the user previously liked
    selected_value = st.selectbox("Select the Course you previously liked", courses['Course Name'].values)
    difficulty_level = st.selectbox("Select the Difficulty Level", courses['Difficulty Level'].unique())

    # Show Recommendations button
    if st.button("Show Recommendations"):
        recommendations = get_course_recommendations(selected_value, difficulty_level, [])

        # Display recommendations and details
        for index, row in recommendations.iterrows():
            # Use HTML styling to highlight the course name with a black background and capitalize it
            st.markdown(f'<div style="background-color: black; padding: 10px; text-align: center; text-transform: uppercase;">{row["Course Name"]}</div>', unsafe_allow_html=True)

            st.write(f"University: {row['University']}")
            st.write(f"Difficulty Level: {row['Difficulty Level']}")
            st.write(f"Course Rating: {row['Course Rating']}")
            st.write(f"Skills: {row['Skills']}")
            st.write(f"Course URL: {row['Course URL']}")
            
            with st.expander(f"Overview"):
               
                course_details = get_course_details(row['Course Name'])
                st.write(course_details['Course Description'])

# Run the Streamlit app
if __name__ == "__main__":
    app()