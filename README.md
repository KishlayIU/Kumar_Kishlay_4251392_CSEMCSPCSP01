# 📌 Project Title - A Retrieval-Augmented Generation Framework for Semantic Video and Audio Search

#  Abstract

With the growing amount of information and knowledge-based audios and videos content available online,
finding certain topics within long recordings is extremely difficult. Traditional search methods that
rely on titles or descriptions often fail to capture the exact moments where a specific concept is
discussed. This project aims to address this difficulty by developing a Retrieval-Augmented Generation
(RAG) system that allows users to search content by topic.

The process begins with extraction of sound from videos using FFmpeg, followed by transcription through
OpenAI Whisper. The generated text is divided into smaller, meaningful chunks and stored in a Pandas-based
database for convenient access and management. When a user enters a query such as “Which file explains HTML
or CSS?”, the system converts the query into a vector, identifies the most relevant text chunks, and uses a
Large Language Model (LLM) to generate a short response with simultaneous file references and timestamps.
 
The goal is to develop a system that allows for easier searching for specific topics within large datasets.
By enabling smarter, context-based retrieval, this tool enhances accessibility, saves time, and makes learning
and content finding more enjoyable.



##  Project Overview

With the rapid growth of informational and knowledge-based video and audio content, finding exactly **where** a particular topic is discussed has become a challenge. Traditional search methods depend only on titles or descriptions, which often fail to locate specific concepts within long recordings.

This project aims to solve that problem by developing a **Retrieval-Augmented Generation (RAG)** system that allows users to **search both videos and audio files by topic**. The system extracts speech, converts it into text, and enables intelligent, topic-based search using **Large Language Models (LLMs)**.



## Key Features  

-  Video-to-Audio Conversion using FFmpeg  
-  Automatic Speech-to-Text Transcription using Whisper  
-  Text Chunking and Semantic Embedding Generation  
-  High-Speed Similarity Search using Cosine Similarity  
-  Retrieval-Augmented Generation (RAG) with LLM  
-  Exact Video Timestamp Retrieval  
-  Supports Both **Audio and Video as Data Sources**  
-  Fully Automated Pipeline  
-  Scalable Vector Storage using `joblib`  
-  Human-Friendly AI Responses




##  Tech Stack



| Category | Tools Used |
|----------|------------|
| Language | Python 3.10+ |
| Video Processing | FFmpeg |
| Speech Recognition | Whisper (large-v2) |
| Embeddings | BGE-M3 (Ollama API) |
| Vector Search | Cosine Similarity |
| LLM | LLaMA 3.2 (Ollama) |
| Data Processing | NumPy, Pandas |
| Storage | JSON, CSV, Joblib |
| ML Utilities | scikit-learn |


##  Quick Start

Follow the steps below to set up and run the project on your local machine:



### 1. Clone the Repository


### 2. Install Dependencies

```bash
pip install -r src/requirements.txt

```

### 3. Run the System
```bash
python src/process_incoming.py
```

##  Folder Structure
```
C:.
├───data
│   ├───audios
│   │       ...audio files...
│   │
│   └───videos
│           ...video files...
│
├───jsons
│       ...JSON transcript files...
│
├───models
│       embeddings.joblib
│
├───output
│       response.txt
│
├───src
│   │   mp3_to_json.py
│   │   preprocess_json.py
│   │   process_incoming.py
│   │   Speech_to_text.py
│   │   video_to_mp3.py
│   │
│   └───llm
│           prompt.txt
│
├───tests
│       test_unrelated_query.py
│       test_valid_query.py
│
├───whisper
├── requirements.txt
└── README.md
   
```        



## System Architecture  

### Core Components  

### 1. Video to Audio Converter (`video_to_mp3.py`)
- Extracts audio from all video files using **FFmpeg**
- Converts videos into `.mp3` format for speech processing
- Stores audio in `/audios` folder



### 2. Speech-to-Text Engine (`mp3_to_json.py`)
- Uses **OpenAI Whisper (large-v2)** model
- Converts speech into text
- Splits output into timestamp-based chunks
- Saves transcription data in `/jsons`



### 3. Embedding Generator (`preprocess_json.py`)
- Converts subtitle chunks into numerical embeddings
- Uses **BGE-M3 embedding model via Ollama**
- Stores embeddings in `embeddings.joblib`



### 4. Query Processing Engine (`process_incoming.py`)
- Accepts a user question
- Generates query embedding
- Applies **Cosine Similarity** with stored embeddings
- Retrieves top 5 most similar subtitle chunks
- Sends selected chunks to LLM for answer generation



### 5. LLM Response Generator (Ollama - LLaMA 3.2)
- Produces human-style answers
- Adds video number and timestamp guidance
- Restricts responses only to course-related queries

##  Example Output Results

```
You're asking about SEO being used in websites. From what I've seen so far, it looks like we covered SEO in some of
our previous videos.

Specifically, in Video 3, titled "Basic Structure of an HTML Website", we talked about how SEO can be used to improve
the structure of a website (start time: 305.72 seconds, end time: 307.44 seconds). We also mentioned that SEO is useful
for search engines like Google to understand the content of a webpage.

Additionally, in Video 6, titled "SEO and Core Web Vitals in HTML", we discussed how to optimize web pages for better
performance and user experience (start time: 63.0 seconds, end time: 66.0 seconds). We also talked about how SEO is used
in websites to improve their visibility in search engine results.

If you want to learn more about how SEO works or how to apply its principles in your own projects, I'd be happy to guide
you further!
```





##  Current Phase Status  
**Phase 4 – System Implementation, Testing & Evaluation Completed**


###  Core System (Complete)

- Audio and video-based content ingestion pipeline
- Video-to-audio conversion using FFmpeg
- Speech-to-text transcription using OpenAI Whisper
- Automatic text chunking with timestamps
- Semantic embedding generation using Ollama (bge-m3 model)
- Pandas-based embedding storage for lightweight retrieval
- Cosine similarity-based semantic search
- LLM-based response generation using LLaMA 3.2
- Timestamp-aware answers with file references
- Modular Python code structure for easy maintenance
- Query-based intelligent search over large media datasets



###  Evaluation Framework (Complete)


- Validation of:
  - Transcription quality
  - Chunk retrieval accuracy
  - Query-to-result mapping
- Retrieval accuracy verified using similarity score ranking
- Output validation through:
  - Generated `prompt.txt` for traceability
  - Generated `response.txt` for final results
- Manual verification of correctness using timestamps


###  Recent Enhancements (Complete)

- Dedicated `models/` directory for storing embeddings
- Dedicated `output/` folder for system responses
- Dynamic absolute path handling for:
  - Embedding model
  - Prompt file
  - Response file
- Improved project folder organization for GitHub deployment




###  Next Phase Goals

- Integration of a vector database (FAISS or ChromaDB)
- Web-based search interface for user queries
- Multi-language transcription and retrieval support
- Enhanced UI with clickable timestamps
- Performance optimization for large-scale datasets



##  Risks

###  Technical Risks
- Limited hardware may slow down transcription and model processing  
- Dependency issues between **FFmpeg**, **Whisper**, and **Python** libraries  

###  Data Risks
- Poor audio quality may reduce transcription accuracy  
- Timestamp mismatches between text and video segments  

###  Performance Risks
- High latency for long videos or large datasets  
- Memory constraints while handling multiple files  

###  Operational Risks
- Internet failure during API calls  
- API key limits or service unavailability  

