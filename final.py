import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
from bertviz import head_view
import streamlit.components.v1 as components

# Set page configuration
st.set_page_config(page_title="Interactive Transformer Visualization", layout="wide")

# Load model and tokenizer once (caching speeds up subsequent runs)
@st.cache(allow_output_mutation=True)
def load_model_and_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

def get_attentions(text_a, text_b=None):
    """
    Given one or two input sentences, this function tokenizes the text,
    computes the model outputs, and returns the attention weights and token list.
    """
    inputs = tokenizer.encode_plus(text_a, text_b, return_tensors='pt')
    outputs = model(**inputs)
    attentions = outputs.attentions  # A tuple of attention tensors (one per layer)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    return attentions, tokens

# Sidebar: Navigation to mimic PPT slides
st.sidebar.title("Interactive Transformer Visualization")
slide = st.sidebar.radio("Select Slide", 
                         ["Introduction", "Default Visualization", "Custom Input Visualization"])

if slide == "Introduction":
    st.title("Interactive Transformer Visualization with BERT")
    st.markdown("""
    Welcome to this interactive presentation that explores how transformer models such as **BERT** work internally.
    
    **In this demo, you can:**
    - Learn about self-attention mechanisms in transformers.
    - See interactive visualizations generated with **BERTViz**.
    - Toggle between a default example and your own custom inputs.
    
    Use the sidebar to move between slides.
    """)

elif slide == "Default Visualization":
    st.title("Default Visualization")
    st.markdown("This slide shows a default example of BERT's attention visualization using a preset sentence pair.")
    
    sentence_a = "The quick brown fox jumps over the lazy dog."
    sentence_b = "The lazy dog was surprised by the quick brown fox."
    
    st.markdown("**Sentence A:** " + sentence_a)
    st.markdown("**Sentence B:** " + sentence_b)
    
    if st.button("Visualize Attention"):
        with st.spinner("Computing attention..."):
            attentions, tokens = get_attentions(sentence_a, sentence_b)
            # Generate the HTML for BERTViz head view
            html = head_view(attentions, tokens)
            # Embed the interactive visualization in Streamlit
            components.html(html, height=800, scrolling=True)

elif slide == "Custom Input Visualization":
    st.title("Custom Input Visualization")
    st.markdown("Enter your own sentences to visualize how BERT's self-attention mechanism responds to different inputs.")
    
    text_a = st.text_area("Enter Sentence A", 
                          value="The quick brown fox jumps over the lazy dog.")
    text_b = st.text_area("Enter Sentence B (optional)", 
                          value="The lazy dog was surprised by the quick brown fox.")
    
    if st.button("Visualize Custom Attention"):
        with st.spinner("Computing attention..."):
            # If Sentence B is left empty, run with a single sentence input.
            attentions, tokens = get_attentions(text_a, text_b if text_b.strip() != "" else None)
            html = head_view(attentions, tokens)
            components.html(html, height=800, scrolling=True)
