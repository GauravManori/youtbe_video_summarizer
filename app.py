#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
from utils import fetch_transcript, split_text_by_tokens, summarize_chunks

st.title("YouTube Video Summarizer")
video_url = st.text_input("Enter YouTube Video URL", placeholder="https://www.youtube.com/watch?v=xxxxxxx")

if st.button("Summarize"):
    if video_url:
        try:
            video_id = video_url.split("v=")[1].split("&")[0]
            st.write("Fetching transcript...")
            transcript = fetch_transcript(video_id)
            if transcript:
                st.write("Transcript fetched successfully. Summarizing...")
                chunks = split_text_by_tokens(transcript, utils.tokenizer, max_tokens=1000)
                long_summary = summarize_chunks(chunks, utils.summarizer)
                st.subheader("Summary:")
                st.write(long_summary)
            else:
                st.error("Could not fetch transcript for the video.")
        except Exception as e:
            st.error(f"Invalid URL or error occurred: {e}")
    else:
        st.error("Please enter a valid YouTube video URL.")

