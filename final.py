import streamlit as st
import streamlit.components.v1 as components
import os
import time

# Set page configuration
st.set_page_config(page_title="Project Presentation for Data Viz final", layout="wide")

# Inject CSS for larger font and fade-in animation for each slide
st.markdown("""
<style>
/* Increase base font size */
body, .markdown-text-container {
    font-size: 20px;
}

/* Fade-in animation */
.fade-in {
    animation: fadeIn 1s ease-in-out;
}
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

/* Optional: Style for the sidebar title and radio buttons if desired */
.sidebar .sidebar-content {
    font-size: 20px;
}
</style>
""", unsafe_allow_html=True)

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

# Function to show a transition effect (requires a transition.gif file in the same directory)
def transition_effect():
    transition_placeholder = st.empty()
    transition_placeholder.image("transition.gif", use_column_width=True)
    time.sleep(0.5)  # Pause briefly for the effect
    transition_placeholder.empty()

# Each slide is wrapped in a container with a fade-in effect
if slide == "Introduction":
    transition_effect()
    with st.container():
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)

elif slide == "Motivation":
    transition_effect()
    with st.container():
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        st.title("Motivation")
        st.markdown(
            """
            **Background & Objectives:**
            
            - In today's data-driven world, understanding the inner workings of transformer models like BERT is essential.
            - We aim to demystify the "black box" of these models by visualizing their attention mechanisms.
            - Improved interpretability can lead to better performance, fairness, and more responsible AI free of bias.
            """
        )
        st.markdown('</div>', unsafe_allow_html=True)

elif slide == "Dataset & Data Wrangling Overview":
    transition_effect()
    with st.container():
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)

elif slide == "Tasks":
    transition_effect()
    with st.container():
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)

elif slide == "Research Questions":
    transition_effect()
    with st.container():
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)

elif slide == "Solutions":
    transition_effect()
    with st.container():
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)

elif slide == "Tutorial":
    transition_effect()
    with st.container():
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)

elif slide == "Results & Findings":
    transition_effect()
    with st.container():
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)

elif slide == "Conclusion":
    transition_effect()
    with st.container():
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)
