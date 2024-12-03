#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
from transformers import pipeline, AutoTokenizer
from youtube_transcript_api import YouTubeTranscriptApi

# Load summarizer and tokenizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

# Function to fetch transcript
def fetch_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t['text'] for t in transcript])
    except Exception as e:
        return None

# Function to split text into chunks based on token limit
def split_text_by_tokens(text, tokenizer, max_tokens=1000):
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

# Function to summarize chunks
def summarize_chunks(chunks, summarizer, max_length=150, min_length=30):
    summaries = []
    for chunk in chunks:
        if chunk.strip():
            input_length = len(chunk.split())
            max_length = min(150, int(input_length * 0.8))
            min_length = max(10, int(input_length * 0.3))
            if max_length <= min_length:
                max_length = min_length + 10
            try:
                summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
                summaries.append(summary[0]['summary_text'])
            except Exception:
                summaries.append(chunk)
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
                chunks = split_text_by_tokens(transcript, tokenizer, max_tokens=1000)
                long_summary = summarize_chunks(chunks, summarizer)

                # Check token limit
                long_summary_tokens = len(tokenizer.encode(long_summary))
                if long_summary_tokens > 1000:
                    summary_chunks = split_text_by_tokens(long_summary, tokenizer, max_tokens=1000)
                    final_summary = summarize_chunks(summary_chunks, summarizer, max_length=100, min_length=80)
                else:
                    final_summary = summarizer(long_summary, max_length=100, min_length=80, do_sample=False)[0]['summary_text']

                st.subheader("Summary:")
                st.write(final_summary)
            else:
                st.error("Could not fetch transcript for the video.")
        except Exception as e:
            st.error(f"Invalid URL or error occurred: {e}")
    else:
        st.error("Please enter a valid YouTube video URL.")

