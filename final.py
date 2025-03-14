import streamlit as st

st.set_page_config(page_title="Project Presentation", layout="wide")

# Sidebar Navigation for Presentation Slides
st.sidebar.title("Project Presentation Slides")
slide = st.sidebar.radio(
    "Select Slide",
    [
        "Motivation",
        "Dataset & Data Wrangling Overview",
        "Tasks",
        "Solutions",
        "Results & Findings",
        "Conclusion"
    ]
)

if slide == "Motivation":
    st.title("Motivation")
    st.markdown(
        """
        **Why This Project?**

        In today’s data-driven world, understanding complex models like BERT is essential for advancing natural language processing.  
        This project was inspired by the need to unravel the “black box” of transformer models and to gain insights into how attention mechanisms work.  

        **Goals:**
        - Enhance model interpretability.
        - Diagnose model behavior using visualization tools.
        - Improve model performance through informed refinements.
        """
    )

elif slide == "Dataset & Data Wrangling Overview":
    st.title("Dataset & Data Wrangling Overview")
    st.markdown(
        """
        **Dataset Overview:**
        - **Source:** Collected from multiple sources (social media, news, academic texts).
        - **Size:** Over 100,000 samples with diverse text data.
        - **Features:** Each sample includes text, metadata, and labels.

        **Data Wrangling Steps:**
        - **Cleaning:** Removal of noise, stopwords, and non-ASCII characters.
        - **Normalization:** Lowercasing, stemming, and lemmatization.
        - **Missing Values:** Imputation techniques applied.
        - **Feature Engineering:** Deriving new features based on domain knowledge.
        """
    )

elif slide == "Tasks":
    st.title("Tasks")
    st.markdown(
        """
        **Project Tasks:**

        1. **Data Exploration:** Analyze the quality and distribution of the dataset.
        2. **Preprocessing:** Clean, normalize, and transform the raw data.
        3. **Modeling:** Fine-tune transformer-based models (e.g., BERT) on the dataset.
        4. **Visualization:** Use tools like BERTViz to visualize internal attention mechanisms.
        5. **Evaluation:** Assess model performance using standard metrics.
        6. **Interpretability:** Use visual insights to diagnose and improve model behavior.
        """
    )

elif slide == "Solutions":
    st.title("Solutions")
    st.markdown(
        """
        **Our Approach:**

        - **Advanced Modeling:** We implemented transformer-based models (BERT) fine-tuned for our specific tasks.
        - **Interactive Visualization:** Leveraged BERTViz to generate interactive visualizations of attention heads and internal representations.
        - **Robust Data Pipeline:** Developed an automated data wrangling and preprocessing pipeline to ensure high-quality inputs.
        - **Interpretability Techniques:** Combined quantitative evaluations with qualitative visual insights to diagnose model performance.
        """
    )

elif slide == "Results & Findings":
    st.title("Results & Findings")
    st.markdown(
        """
        **Key Results:**

        - **Improved Accuracy:** Achieved an 8-12% boost in model accuracy compared to baseline approaches.
        - **Attention Insights:** Visualizations revealed that certain attention heads capture syntactic patterns while others focus on semantic relationships.
        - **Data Insights:** Uncovered hidden correlations between metadata and text sentiment.
        - **Model Diagnostics:** Visualization-driven insights guided further fine-tuning and error analysis.

        **Conclusions:**

        - Transformer models are not only powerful in performance but also offer avenues for enhanced interpretability.
        - Visual diagnostic tools like BERTViz can significantly improve our understanding of model behavior.
        """
    )

elif slide == "Conclusion":
    st.title("Conclusion")
    st.markdown(
        """
        **Summary:**

        - This project demonstrates how advanced transformer models can be made interpretable through visualization.
        - Thorough data wrangling and exploration are critical to achieving robust model performance.
        - Interactive visualization tools bridge the gap between model performance and transparency.

        **Future Work:**
        
        - Scale the analysis to larger, more diverse datasets.
        - Experiment with additional transformer architectures and interpretability methods.
        - Incorporate user feedback to further refine model diagnostics and improvements.

        **Thank you for your attention!**
        """
    )
