import json
import cv2
import streamlit as st
import os
import numpy as np
from PIL import Image
from htr_pipeline import read_page, DetectorConfig, LineClusteringConfig, ReaderConfig, PrefixTree

# Set page config for a cleaner look
st.set_page_config(
    page_title="Handwritten Text Recognition",
    page_icon="✍️",
    layout="wide"
)

# Change working directory to script location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Initialize the word list and prefix tree
@st.cache_resource  # Cache this resource to avoid reloading
def initialize_word_list():
    with open('../data/words_alpha.txt') as f:
        word_list = [w.strip().upper() for w in f.readlines()]
    return PrefixTree(word_list)

prefix_tree = initialize_word_list()

def process_page(img, scale, margin, use_dictionary, min_words_per_line, text_scale):
    # Convert PIL Image to numpy array if necessary
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    # read page
    with st.spinner('Processing image...'):
        read_lines = read_page(img,
                            detector_config=DetectorConfig(scale=scale, margin=margin),
                            line_clustering_config=LineClusteringConfig(min_words_per_line=min_words_per_line),
                            reader_config=ReaderConfig(decoder='word_beam_search' if use_dictionary else 'best_path',
                                                    prefix_tree=prefix_tree))

    # create text to show
    res = ''
    for read_line in read_lines:
        res += ' '.join(read_word.text for read_word in read_line) + '\n'

    # create visualization to show
    img_viz = img.copy()
    for i, read_line in enumerate(read_lines):
        for read_word in read_line:
            aabb = read_word.aabb
            cv2.rectangle(img_viz,
                        (aabb.xmin, aabb.ymin),
                        (aabb.xmin + aabb.width, aabb.ymin + aabb.height),
                        (255, 0, 0),
                        2)
            cv2.putText(img_viz,
                      read_word.text,
                      (aabb.xmin, aabb.ymin + aabb.height // 2),
                      cv2.FONT_HERSHEY_SIMPLEX,
                      text_scale,
                      color=(255, 0, 0))

    return res, img_viz

def main():
    st.title("✍️ Handwritten Text Recognition")
    st.write("Upload an image containing handwritten text to process")

    # Create two columns for parameters
    col1, col2 = st.columns(2)

    with col1:
        scale = st.slider("Scale", 0.0, 10.0, 1.0, 0.01)
        margin = st.slider("Margin", 0, 25, 1, 1)
        use_dictionary = st.checkbox("Use Dictionary", value=False)

    with col2:
        min_words_per_line = st.slider("Minimum Words per Line", 1, 10, 2, 1)
        text_scale = st.slider("Text Size in Visualization", 0.5, 2.0, 1.0, 0.1)

    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

        # Process button
        if st.button("Process Image"):
            try:
                # Process the image
                recognized_text, visualized_image = process_page(
                    image, scale, margin, use_dictionary, 
                    min_words_per_line, text_scale
                )

                # Create columns for results
                text_col, viz_col = st.columns(2)

                with text_col:
                    st.subheader("Recognized Text")
                    st.text_area("", recognized_text, height=200)

                with viz_col:
                    st.subheader("Visualization")
                    st.image(visualized_image, use_column_width=True)

                # Add download button for the text
                st.download_button(
                    label="Download Recognized Text",
                    data=recognized_text,
                    file_name="recognized_text.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Add example images section
    st.subheader("Or try with example images")
    try:
        with open('../data/config.json') as f:
            config = json.load(f)
            
        # Create a grid of example images
        cols = st.columns(len(config))
        for idx, (k, v) in enumerate(config.items()):
            with cols[idx]:
                example_path = os.path.abspath(os.path.join('..', 'data', k))
                if os.path.exists(example_path):
                    st.image(example_path, caption=f"Example {idx+1}")
                    if st.button(f"Use Example {idx+1}", key=f"example_{idx}"):
                        image = Image.open(example_path)
                        recognized_text, visualized_image = process_page(
                            image, v['scale'], v['margin'], 
                            False, 2, v['text_scale']
                        )
                        
                        text_col, viz_col = st.columns(2)
                        with text_col:
                            st.subheader("Recognized Text")
                            st.text_area("", recognized_text, height=200, key=f"text_{idx}")
                        
                        with viz_col:
                            st.subheader("Visualization")
                            st.image(visualized_image, use_column_width=True)
    except Exception as e:
        st.warning(f"Could not load example images: {str(e)}")

    # Add footer
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit")

if __name__ == "__main__":
    main()
