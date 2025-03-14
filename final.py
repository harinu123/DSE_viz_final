import streamlit as st
import streamlit.components.v1 as components
import os

# Set page configuration
st.set_page_config(page_title="Project Presentation with BERTViz", layout="wide")

# Define slide options in the desired order
slides = [
    "Introduction",
    "Motivation",
    "Dataset & Data Wrangling Overview",
    "Tasks",
    "Research Questions",
    "Solutions",
    "Tutorial",
    "Results & Findings",
    "Conclusion"
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
        
        This presentation is divided into two main parts:
        
        1. **Project Overview:** Background, motivation, dataset preparation, tasks, research questions, solutions, and results.
        2. **Tutorial:** A complete interactive walkthrough (pre-generated) showing our results and visualizations.
        
        Use the sidebar to navigate between slides.
        """
    )

# Slide: Motivation
elif slide == "Motivation":
    st.title("Motivation")
    st.markdown(
        """
        **Background & Objectives:**
        
        - In today's data-driven world, understanding the inner workings of transformer models like BERT is essential.
        - We aim to demystify the "black box" of these models by visualizing their attention mechanisms.
        - Improved interpretability can lead to better performance, fairness, and more responsible AI free of bias.
        """
    )

# Slide: Dataset & Data Wrangling Overview
elif slide == "Dataset & Data Wrangling Overview":
    st.title("Dataset & Data Wrangling Overview")
    st.markdown(
        """
        **Dataset Overview:**
        - **Source:** BERT model from hugging face.
        - **Size:** 300 million parameter model 

        
        **Preprocessing & Augmentation:**
        - **Setup:** We load the tokenizer and the attention parts. No further modification to the model.
        """
    )

# Slide: Tasks
elif slide == "Tasks":
    st.title("Project Tasks")
    st.markdown(
        """
        **Key Tasks:**
        
        1. **Data Exploration:** Analyze and understand the dataset.
        2. **Preprocessing:** Clean and transform the raw data.
        3. **Modeling:** Fine-tune transformer-based models like BERT.
        4. **Visualization:** Use BERTViz to inspect internal attention mechanisms.
        5. **Evaluation:** Assess model performance using standard metrics.
        6. **Interpretability:** Leverage visual insights to diagnose and improve model behavior.
        """
    )

# Slide: Research Questions
elif slide == "Research Questions":
    st.title("Research Questions: Investigating Gender Bias via Attention")
    st.markdown(
        """
        We aim to explore whether the language model exhibits gender bias using its attention mechanisms.  
        
        **Five Key Questions:**
        
        1. **Gendered Token Attention:**  
           How do attention patterns differ when processing sentences with gender-specific pronouns (e.g., "he" vs. "she") in similar contexts?
           
        2. **Role and Occupation Bias:**  
           When gendered pronouns are combined with occupation-related terms (e.g., "nurse" vs. "doctor"), how is attention distributed between these tokens?
           
        3. **Contextual Dependency Effects:**  
           Does the presence of gendered language shift attention to surrounding context words, affecting semantic interpretation?
           
        4. **Layer-wise and Head-wise Variability:**  
           At what depth (layer or head) do gender-specific patterns emerge? Are lower layers focused on syntax and higher layers on semantics?
           
        5. **Attention-Driven Prediction Bias:**  
           Can variations in attention patterns for gendered sentences be linked to differences in downstream predictions, indicating a real-world bias?
        """
    )

# Slide: Solutions
elif slide == "Solutions":
    st.title("Solutions")
    st.markdown(
        """
        **Our Approach:**
        
        - Implemented a transformer-based model (BERT) fine-tuned on our dataset.
        - Leveraged interactive visualization tools to explore attention mechanisms.
        - Designed controlled experiments (e.g., minimal pairs) to test for gender bias.
        - Combined qualitative visualization insights with quantitative attention analysis.
        """
    )

# Slide: Tutorial (Embedding the full HTML file)
elif slide == "Tutorial":
    st.title("Tutorial")
    st.markdown(
        """
        Below is the complete interactive tutorial that includes all our results and visualizations.
        This pre-generated HTML file contains interactive elements from our analysis.
        """
    )
    html_code = load_html("tutorial.html")
    st.write(f"Loaded tutorial.html (length: {len(html_code)} characters)")
    components.html(html_code, height=800, scrolling=True)

# Slide: Results & Findings
elif slide == "Results & Findings":
    st.title("Results & Findings")
    st.markdown(
        """
        **Key Outcomes:**
        
        - Visualizations revealed that some attention heads capture syntactic structure while others capture semantics.
        - Experiments with minimal pairs indicate differences in attention patterns for gendered tokens, suggesting potential bias.
        - Quantitative analysis of attention weights supports these visual insights.
        
        **Conclusions:**
        - Enhanced interpretability and diagnosis of transformer models.
        - Identification of specific areas for further bias mitigation and model improvement.
        """
    )

# Slide: Conclusion
elif slide == "Conclusion":
    st.title("Conclusion")
    st.markdown(
        """
        **Summary:**
        
        - This project demonstrates how advanced transformer models like BERT can be interpreted through visualization.
        - A well-designed data pipeline and interactive visualization are key to understanding model behavior.
        - Our investigation into gender bias via attention mechanisms offers valuable insights for improving model fairness.
        
        **Future Work:**
        - Scale the analysis to larger, more diverse datasets.
        - Explore additional transformer architectures and interpretability techniques.
        - Integrate user feedback to refine models and bias mitigation strategies.
        
        **Thank you for your attention!**
        """
    )
