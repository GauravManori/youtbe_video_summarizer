#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
from youtube_transcript_api import YouTubeTranscriptApi

# Load T5-small model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Function to fetch transcript
def fetch_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t['text'] for t in transcript])
    except Exception as e:
        return None

# Function to split text into chunks based on token limit
def split_text_by_tokens(text, tokenizer, max_tokens=512):
    tokens = tokenizer.encode(text, return_tensors="pt")[0]
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

# Function to summarize text using T5-small
def summarize_text(chunks, model, tokenizer, max_length=150, min_length=30):
    summaries = []
    for chunk in chunks:
        if chunk.strip():
            input_ids = tokenizer.encode("summarize: " + chunk, return_tensors="pt", max_length=512, truncation=True)
            output = model.generate(input_ids, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
            summaries.append(tokenizer.decode(output[0], skip_special_tokens=True))
    return " ".join(summaries)

# Streamlit interface
st.title("YouTube Video Summarizer")
video_url = st.text_input("Enter YouTube Video URL", placeholder="https://www.youtube.com/watch?v=xxxxxxx")

if st.button("Summarize"):
    if video_url:
        # Extract video ID from URL
        try:
            video_id = video_url.split("v=")[1].split("&")[0]
            st.write("Fetching transcript...")
            transcript = fetch_transcript(video_id)
            if transcript:
                st.write("Transcript fetched successfully. Summarizing...")
                chunks = split_text_by_tokens(transcript, tokenizer, max_tokens=512)
                summary = summarize_text(chunks, model, tokenizer)
                st.subheader("Summary:")
                st.write(summary)
            else:
                st.error("Could not fetch transcript for the video.")
        except Exception as e:
            st.error(f"Invalid URL or error occurred: {e}")
    else:
        st.error("Please enter a valid YouTube video URL.")

