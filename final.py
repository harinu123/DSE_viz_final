import streamlit as st
import asyncio
import torch
import os

# ----------------------------------------------------------------
# Hack to suppress PyTorch's __path__._path error
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]
# ----------------------------------------------------------------

# Ensure a running event loop
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

from IPython.display import display, HTML, Javascript  # recommended import

from transformers import AutoTokenizer, BertModel
from bertviz import head_view
import streamlit.components.v1 as components

st.set_page_config(page_title="Interactive Transformer Visualization", layout="wide")

@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained(
        'bert-base-uncased',
        output_attentions=True,
        attn_implementation="eager"
    )
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

def get_attentions(text_a, text_b=None):
    inputs = tokenizer.encode_plus(text_a, text_b, return_tensors='pt')
    outputs = model(**inputs)
    attentions = outputs.attentions
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    return attentions, tokens

st.sidebar.title("Interactive Transformer Visualization")
slide = st.sidebar.radio("Select Slide", 
                         ["Introduction", "Default Visualization", "Custom Input Visualization"])

if slide == "Introduction":
    st.title("Interactive Transformer Visualization with BERT")
    st.markdown(
        """
        Welcome! This app visualizes the self-attention patterns of a BERT model.
        
        **Slides**:
        1. Introduction
        2. Default Visualization
        3. Custom Input Visualization
        
        Use the sidebar to navigate.
        """
    )

elif slide == "Default Visualization":
    st.title("Default Visualization")
    sentence_a = "The quick brown fox jumps over the lazy dog."
    sentence_b = "The lazy dog was surprised by the quick brown fox."

    st.markdown("**Sentence A:** " + sentence_a)
    st.markdown("**Sentence B:** " + sentence_b)

    if st.button("Visualize Attention"):
        with st.spinner("Computing attention..."):
            attentions, tokens = get_attentions(sentence_a, sentence_b)
            # Find the first [SEP] to separate sentence A from sentence B
            sep_idx = tokens.index('[SEP]')
            # Generate the HTML
            html_code = head_view(attentions, tokens, sentence_b_start=sep_idx)
            
            # Debug: Show how many characters the HTML has
            st.write("HTML code length:", len(html_code))
            
            # 1) Attempt to render via components.html
            components.html(html_code, height=800, scrolling=True)
            
            # 2) Also attempt to render via st.markdown (comment out if not needed)
            # st.markdown(html_code, unsafe_allow_html=True)

elif slide == "Custom Input Visualization":
    st.title("Custom Input Visualization")
    text_a = st.text_area("Enter Sentence A", value="The quick brown fox jumps over the lazy dog.")
    text_b = st.text_area("Enter Sentence B (optional)", value="The lazy dog was surprised by the quick brown fox.")

    if st.button("Visualize Custom Attention"):
        with st.spinner("Computing attention..."):
            # If user provided second sentence
            if text_b.strip():
                attentions, tokens = get_attentions(text_a, text_b)
                sep_idx = tokens.index('[SEP]')
                html_code = head_view(attentions, tokens, sentence_b_start=sep_idx)
            else:
                # Single sentence
                attentions, tokens = get_attentions(text_a)
                html_code = head_view(attentions, tokens)

            st.write("HTML code length:", len(html_code))
            components.html(html_code, height=800, scrolling=True)
            st.markdown(html_code, unsafe_allow_html=True)
