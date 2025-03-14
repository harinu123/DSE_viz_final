import streamlit as st
import asyncio
import torch
import os

# Suppress PyTorch's __path__._path error
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

# Ensure a running event loop
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# BERTViz might call display(...) directly, so we monkey-patch
import IPython.display as ipd
from IPython.display import HTML, Javascript

from transformers import AutoTokenizer, BertModel
from bertviz import head_view
import streamlit.components.v1 as components

st.set_page_config(page_title="Interactive Transformer Visualization", layout="wide")

###############################################################################
# 1) Monkey Patch: capture_bertviz_html
###############################################################################
def capture_bertviz_html(fn, *args, **kwargs):
    """
    If head_view() calls display(HTML(...)) or display(Javascript(...)) and returns None,
    this function overrides IPython.display.display to capture those calls as a single
    HTML/JS string.
    """
    captured = []

    def my_display(*objs, **disp_kwargs):
        for obj in objs:
            if isinstance(obj, HTML):
                captured.append(obj.data)
            elif isinstance(obj, Javascript):
                captured.append(obj.data)
            else:
                # fallback if some other object is displayed
                captured.append(str(obj))

    original_display = ipd.display
    ipd.display = my_display
    try:
        fn(*args, **kwargs)  # e.g., head_view(...)
    finally:
        ipd.display = original_display

    return "\n".join(captured)

###############################################################################
# 2) Load model & tokenizer with st.cache_resource
###############################################################################
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained(
        'bert-base-uncased',
        output_attentions=True,
        attn_implementation="eager"  # Use manual attention to avoid warnings
    )
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

###############################################################################
# 3) Function to get attentions & tokens
###############################################################################
def get_attentions(text_a, text_b=None):
    inputs = tokenizer.encode_plus(text_a, text_b, return_tensors='pt')
    outputs = model(**inputs)
    attentions = outputs.attentions  # tuple of attention tensors
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    return attentions, tokens

###############################################################################
# 4) Streamlit "Slides"
###############################################################################
st.sidebar.title("Interactive Transformer Visualization")
slide = st.sidebar.radio(
    "Select Slide", 
    ["Introduction", "Default Visualization", "Custom Input Visualization"]
)

if slide == "Introduction":
    st.title("Interactive Transformer Visualization with BERT")
    st.markdown(
        """
        Welcome to this interactive presentation that explores how transformer models 
        such as **BERT** work internally.
        
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
            # Find index of first [SEP] to mark boundary between Sentence A & B
            try:
                sep_idx = tokens.index('[SEP]')
            except ValueError:
                sep_idx = None

            # Capture the HTML from BERTViz
            html_code = capture_bertviz_html(
                head_view,
                attentions,
                tokens,
                sentence_b_start=sep_idx
            )

            st.write("HTML code length:", len(html_code))
            if html_code:
                components.html(html_code, height=800, scrolling=True)
            else:
                st.error("No HTML output was captured from BERTViz. Check versions/compatibility.")

elif slide == "Custom Input Visualization":
    st.title("Custom Input Visualization")
    text_a = st.text_area("Enter Sentence A", value="The quick brown fox jumps over the lazy dog.")
    text_b = st.text_area("Enter Sentence B (optional)", value="The lazy dog was surprised by the quick brown fox.")
    
    if st.button("Visualize Custom Attention"):
        with st.spinner("Computing attention..."):
            if text_b.strip():
                attentions, tokens = get_attentions(text_a, text_b)
                try:
                    sep_idx = tokens.index('[SEP]')
                except ValueError:
                    sep_idx = None
            else:
                attentions, tokens = get_attentions(text_a)
                sep_idx = None

            html_code = capture_bertviz_html(
                head_view,
                attentions,
                tokens,
                sentence_b_start=sep_idx
            )

            st.write("HTML code length:", len(html_code))
            if html_code:
                components.html(html_code, height=800, scrolling=True)
            else:
                st.error("No HTML output was captured from BERTViz. Check versions/compatibility.")
