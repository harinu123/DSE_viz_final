import streamlit as st
import asyncio
import torch
import os

# -----------------------------
# Hack to suppress PyTorch's __path__._path error:
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]
# -----------------------------

# Ensure a running event loop (needed for some Streamlit setups)
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Use the recommended import for IPython.display to avoid deprecation warnings
from IPython.display import display, HTML, Javascript

from transformers import AutoTokenizer, BertModel
from bertviz import head_view
import streamlit.components.v1 as components

st.set_page_config(page_title="Interactive Transformer Visualization", layout="wide")

# Use st.cache_resource to cache expensive resources (model & tokenizer)
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained(
        'bert-base-uncased',
        output_attentions=True,
        attn_implementation="eager"  # avoid the manual attention warning
    )
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

def get_attentions(text_a, text_b=None):
    """
    Encode one or two sentences and return (attentions, tokens).
    """
    inputs = tokenizer.encode_plus(text_a, text_b, return_tensors='pt')
    outputs = model(**inputs)
    attentions = outputs.attentions  # tuple of (layers x batch x heads x seq_len x seq_len)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    return attentions, tokens

# Sidebar navigation to simulate "slides"
st.sidebar.title("Interactive Transformer Visualization")
slide = st.sidebar.radio(
    "Select Slide", 
    ["Introduction", "Default Visualization", "Custom Input Visualization"]
)

if slide == "Introduction":
    st.title("Interactive Transformer Visualization with BERT")
    st.markdown(
        """
        Welcome to this interactive presentation that explores how transformer models such as **BERT** work internally.
        
        **In this demo, you can:**
        - Learn about self-attention mechanisms in transformers.
        - See interactive visualizations generated with **BERTViz**.
        - Toggle between a default example and your own custom inputs.
        
        Use the sidebar to move between slides.
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
            # Identify where the first sentence ends so BERTViz can split them
            sep_idx = tokens.index('[SEP]')  # first [SEP]
            html = head_view(
                attentions,
                tokens,
                sentence_b_start=sep_idx
            )
            components.html(html, height=800, scrolling=True)

elif slide == "Custom Input Visualization":
    st.title("Custom Input Visualization")
    text_a = st.text_area("Enter Sentence A", value="The quick brown fox jumps over the lazy dog.")
    text_b = st.text_area("Enter Sentence B (optional)", value="The lazy dog was surprised by the quick brown fox.")
    
    if st.button("Visualize Custom Attention"):
        with st.spinner("Computing attention..."):
            # If the user provided a second sentence
            if text_b.strip():
                attentions, tokens = get_attentions(text_a, text_b)
                sep_idx = tokens.index('[SEP]')
                html = head_view(attentions, tokens, sentence_b_start=sep_idx)
            else:
                # Single-sentence mode
                attentions, tokens = get_attentions(text_a)
                html = head_view(attentions, tokens)
            
            components.html(html, height=800, scrolling=True)
