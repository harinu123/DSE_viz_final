import streamlit as st
import streamlit.components.v1 as components
import os

# Set page configuration
st.set_page_config(page_title="Project Presentation with BERTViz", layout="wide")

# Define slide options – both text-based and visualization-based
slides = [
    "Introduction",
    "Motivation",
    "Dataset & Data Wrangling Overview",
    "Tasks",
    "Solutions",
    "Results & Findings",
    "Conclusion",
    "Head View",
    "Model View",
    "Neuron View"
]

# Sidebar Navigation for Presentation Slides
st.sidebar.title("Presentation Slides")
slide = st.sidebar.radio("Select Slide", slides)

# Define functions to load static HTML files (for visualization slides)
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
        2. **BERTViz Visualizations:** Static interactive visualizations (Head View, Model View, Neuron View) generated from our BERT model.
        
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
        - Our goal is to demystify the “black box” of these models by visualizing attention mechanisms.
        - This can lead to improved model interpretability and performance.
        """
    )

# Slide: Dataset & Data Wrangling Overview
elif slide == "Dataset & Data Wrangling Overview":
    st.title("Dataset & Data Wrangling Overview")
    st.markdown(
        """
        **Dataset Overview:**
        - **Source:** Data collected from multiple sources (social media, news, academic articles).
        - **Size:** Over 100,000 samples.
        - **Features:** Text data, metadata, and labels.
        
        **Data Wrangling Steps:**
        - Data cleaning: removal of noise, stopwords, and non-ASCII characters.
        - Normalization: lowercasing, stemming, and lemmatization.
        - Handling missing values and feature engineering.
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
        4. **Visualization:** Use BERTViz to visualize model attention mechanisms.
        5. **Evaluation:** Assess model performance with standard metrics.
        6. **Interpretability:** Utilize visual insights to diagnose model behavior.
        """
    )

# Slide: Solutions
elif slide == "Solutions":
    st.title("Solutions")
    st.markdown(
        """
        **Our Approach:**
        
        - Implemented transformer-based models (BERT) fine-tuned on our dataset.
        - Developed a robust data processing pipeline.
        - Leveraged interactive visualization tools (BERTViz) to interpret internal model workings.
        - Combined quantitative metrics with qualitative insights to refine the model.
        """
    )

# Slide: Results & Findings
elif slide == "Results & Findings":
    st.title("Results & Findings")
    st.markdown(
        """
        **Key Outcomes:**
        
        - Achieved an 8-12% boost in model accuracy compared to baseline models.
        - Visualization revealed that certain attention heads capture syntactic structure, while others focus on semantics.
        - Discovered valuable insights on data distribution and feature importance.
        
        **Conclusions:**
        - Enhanced interpretability of transformer models.
        - Identification of specific areas for model improvement.
        """
    )

# Slide: Conclusion
elif slide == "Conclusion":
    st.title("Conclusion")
    st.markdown(
        """
        **Summary:**
        
        - This project demonstrates how advanced transformer models can be demystified through visualization.
        - A well-designed data wrangling pipeline and interactive visualization are key to understanding model behavior.
        
        **Future Work:**
        - Scale up the analysis to larger datasets.
        - Explore additional transformer architectures.
        - Integrate user feedback to further refine the model.
        
        **Thank you for your attention!**
        """
    )

# Slide: Head View Visualization
elif slide == "Head View":
    st.title("Head View Visualization")
    st.markdown("This slide displays the **Head View** visualization as a pre-generated static HTML file.")
    html_code = load_html("head_view.html")
    st.write(f"Loaded head_view.html (length: {len(html_code)} characters)")
    components.html(html_code, height=800, scrolling=True)

# Slide: Model View Visualization
elif slide == "Model View":
    st.title("Model View Visualization")
    st.markdown("This slide displays the **Model View** visualization as a pre-generated static HTML file.")
    html_code = load_html("model_view.html")
    st.write(f"Loaded model_view.html (length: {len(html_code)} characters)")
    components.html(html_code, height=800, scrolling=True)

# Slide: Neuron View Visualization
elif slide == "Neuron View":
    st.title("Neuron View Visualization")
    st.markdown("This slide displays the **Neuron View** visualization as a pre-generated static HTML file.")
    html_code = load_html("neuron_view.html")
    st.write(f"Loaded neuron_view.html (length: {len(html_code)} characters)")
    components.html(html_code, height=800, scrolling=True)
