import streamlit as st
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# Function to generate a caption
def generate_caption(image):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# Function to find the most similar emoji
def find_similar_emoji(caption, sentence_model, emoji_df):
    caption_embedding = sentence_model.encode([caption])[0]
    caption_embedding = np.array(caption_embedding).reshape(1, -1)
    
    emoji_embeddings = {row['emoji']: np.array(sentence_model.encode([row['name']])[0]).reshape(1, -1) for _, row in emoji_df.iterrows()}
    
    similarities = {emoji: cosine_similarity(caption_embedding, emoji_embedding)[0][0] for emoji, emoji_embedding in emoji_embeddings.items()}
    return max(similarities, key=similarities.get)

# Streamlit app
def main():
    st.title("Image to Emoji Converter")
    
    # User uploads an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        # Display the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        # Load the sentence transformer model
        sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Load emoji data
        emoji_df = pd.read_csv("./emoji_df.csv")  # Ensure the file is in the same directory as the app

        # Generate caption
        caption = generate_caption(image)
        st.write("Caption:", caption)

        # Find and display the most similar emoji
        emoji = find_similar_emoji(caption, sentence_model, emoji_df)
        st.write("Most similar emoji: ", emoji)

if __name__ == "__main__":
    main()
