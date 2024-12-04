#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import pipeline, AutoTokenizer
from youtube_transcript_api import YouTubeTranscriptApi

# Initialize summarizer and tokenizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

def fetch_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t['text'] for t in transcript])
    except Exception as e:
        return None

def split_text_by_tokens(text, tokenizer, max_tokens=1000):
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

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

