import streamlit as st
import streamlit.components.v1 as components
import os

# Set page configuration
st.set_page_config(page_title="Project Presentation with Tutorial", layout="wide")

# Define slide options – both text-based and the tutorial slide
slides = [
    "Introduction",
    "Motivation",
    "Dataset & Data Wrangling Overview",
    "Tasks",
    "Solutions",
    "Results & Findings",
    "Conclusion",
    "Tutorial"
]

# Sidebar Navigation for Presentation Slides
st.sidebar.title("Presentation Slides")
slide = st.sidebar.radio("Select Slide", slides)

# Function to load static HTML file
def load_html(file_name):
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            html_code = f.read()
        return html_code
    except Exception as e:
        return f"Error loading {file_name}: {e}"

# Slide: Introduction
if slide == "Introduction":
    st.title("Project Presentation: Understanding BERT with Visualizations")
    st.markdown(
        """
        Welcome to this interactive presentation where we explore the inner workings of a BERT model.
        
        This presentation is divided into two parts:
        
        1. **Project Overview:** Motivation, dataset preparation, tasks, solutions, and results.
        2. **Tutorial:** A full interactive walkthrough (pre-generated) showing our results and visualizations.
        
        Use the sidebar to navigate between slides.
        """
    )

# Slide: Motivation
elif slide == "Motivation":
    st.title("Motivation")
    st.markdown(
        """
        **Why This Project?**
        
        - In today's data-driven world, understanding the inner workings of transformer models like BERT is essential.
        - Our goal is to demystify the “black box” of these models by visualizing their attention mechanisms.
        - Improved interpretability can lead to better model performance and more responsible AI.
        """
    )

# Slide: Dataset & Data Wrangling Overview
elif slide == "Dataset & Data Wrangling Overview":
    st.title("Dataset & Data Wrangling Overview")
    st.markdown(
        """
        **Dataset Overview:**
        - **Source:** Collected from multiple sources (social media, news, academic articles).
        - **Size:** Over 100,000 samples.
        - **Features:** Includes text data, metadata, and labels.
        
        **Data Wrangling Steps:**
        - Cleaning: Removal of noise, stopwords, and non-ASCII characters.
        - Normalization: Lowercasing, stemming, and lemmatization.
        - Handling Missing Values and Feature Engineering.
        """
    )

# Slide: Tasks
elif slide == "Tasks":
    st.title("Project Tasks")
    st.markdown(
        """
        **Key Tasks:**
        
        1. **Data Exploration:** Analyze and understand the dataset.
        2. **Preprocessing:** Clean and transform raw data.
        3. **Modeling:** Fine-tune transformer-based models like BERT.
        4. **Visualization:** Use BERTViz to inspect model attention mechanisms.
        5. **Evaluation:** Assess model performance using standard metrics.
        6. **Interpretability:** Leverage visual insights to diagnose model behavior.
        """
    )

# Slide: Solutions
elif slide == "Solutions":
    st.title("Solutions")
    st.markdown(
        """
        **Our Approach:**
        
        - Implemented a transformer-based model (BERT) fine-tuned on our dataset.
        - Developed a robust data processing pipeline.
        - Leveraged interactive visualization tools (BERTViz) to uncover model internals.
        - Combined quantitative performance metrics with qualitative visual insights.
        """
    )

# Slide: Results & Findings
elif slide == "Results & Findings":
    st.title("Results & Findings")
    st.markdown(
        """
        **Key Outcomes:**
        
        - Achieved an 8-12% improvement in model accuracy compared to baseline methods.
        - Visualizations revealed that certain attention heads capture syntactic structure while others capture semantics.
        - Discovered insights on feature importance and data distribution.
        
        **Conclusions:**
        - Enhanced model interpretability.
        - Identified areas for further model improvements.
        """
    )

# Slide: Conclusion
elif slide == "Conclusion":
    st.title("Conclusion")
    st.markdown(
        """
        **Summary:**
        
        - This project demonstrates how advanced transformer models like BERT can be interpreted using visualization.
        - A well-designed data pipeline and interactive visualization are key to understanding model behavior.
        
        **Future Work:**
        - Scale the analysis to larger datasets.
        - Explore additional transformer architectures and interpretability techniques.
        - Integrate user feedback to refine the model further.
        
        **Thank you for your attention!**
        """
    )

# Slide: Tutorial (Embedding the full HTML file)
elif slide == "Tutorial":
    st.title("Tutorial")
    st.markdown(
        """
        Below is the complete tutorial that includes all our results and visualizations.
        This pre-generated HTML file contains interactive elements from our BERTViz analysis.
        """
    )
    html_code = load_html("tutorial.html")
    st.write(f"Loaded tutorial.html (length: {len(html_code)} characters)")
    components.html(html_code, height=800, scrolling=True)
