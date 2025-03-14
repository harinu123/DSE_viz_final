import streamlit as st
import asyncio
import torch
import os

# ----------------------------------------------------------------
# Hack to suppress PyTorch's __path__._path error:
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]
# ----------------------------------------------------------------

# Ensure a running event loop (needed in some Streamlit setups)
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Use the recommended import for IPython.display (avoids deprecation warnings)
from IPython.display import HTML, Javascript
import IPython.display as ipyd

from transformers import AutoTokenizer, BertModel
from bertviz import head_view
import streamlit.components.v1 as components

st.set_page_config(page_title="Interactive Transformer Visualization", layout="wide")

# Cache the expensive resources using st.cache_resource
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained(
        'bert-base-uncased',
        output_attentions=True,
        attn_implementation="eager"  # Use manual implementation to avoid warnings
    )
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

def get_attentions(text_a, text_b=None):
    """
    Tokenizes one or two sentences and returns the attention tensors and token list.
    """
    inputs = tokenizer.encode_plus(text_a, text_b, return_tensors='pt')
    outputs = model(**inputs)
    attentions = outputs.attentions  # tuple of attention tensors
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    return attentions, tokens

def capture_display(func, *args, **kwargs):
    """
    Captures the output of a display call.
    head_view normally calls display() (which outputs IPython.display.HTML/Javascript objects)
    but returns None. This helper temporarily overrides display() to capture the HTML string.
    """
    captured = []

    def my_display(x):
        if isinstance(x, HTML):
            captured.append(x.data)
        elif isinstance(x, Javascript):
            captured.append(x.data)
        else:
            captured.append(str(x))

    original_display = ipyd.display
    ipyd.display = my_display
    try:
        func(*args, **kwargs)
    finally:
        ipyd.display = original_display
    return "".join(captured)

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
            # Find the index of the first [SEP] to mark the boundary between sentences
            sep_idx = tokens.index('[SEP]')
            # Capture the HTML output from head_view by overriding display()
            html_code = capture_display(head_view, attentions, tokens, sentence_b_start=sep_idx)
            st.write("HTML code length:", len(html_code))
            components.html(html_code, height=800, scrolling=True)

elif slide == "Custom Input Visualization":
    st.title("Custom Input Visualization")
    text_a = st.text_area("Enter Sentence A", value="The quick brown fox jumps over the lazy dog.")
    text_b = st.text_area("Enter Sentence B (optional)", value="The lazy dog was surprised by the quick brown fox.")
    
    if st.button("Visualize Custom Attention"):
        with st.spinner("Computing attention..."):
            if text_b.strip():
                attentions, tokens = get_attentions(text_a, text_b)
                sep_idx = tokens.index('[SEP]')
                html_code = capture_display(head_view, attentions, tokens, sentence_b_start=sep_idx)
            else:
                attentions, tokens = get_attentions(text_a)
                html_code = capture_display(head_view, attentions, tokens)
            st.write("HTML code length:", len(html_code))
            components.html(html_code, height=800, scrolling=True)
