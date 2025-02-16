# Video Semantic Search Application

This project allows users to upload a video, transcribe its content using OpenAI’s Whisper, and perform semantic searches over the transcript using a Sentence Transformer model. I used Python Django for the backend and Next.js for the frontend. Containerization is achieved using Docker, and Poetry is used for managing Python dependencies.

## Features

### Video Upload & Transcription
- **Upload:** The backend accepts video uploads.
- **Transcription:** OpenAI’s Whisper generates a transcription and timestamped segments.
- **Frame Extraction & Captioning:**  
  - OpenCV extracts frames from the video.
  - An image captioning model (BLIP) generates captions for the frames.
- **Data Storage:**  
  - Transcripts and segments are stored in the database.
  - Frame captions are stored alongside the transcript segments.

### Semantic Search
- **Embeddings:**  
  - The transcript is encoded using the SentenceTransformer model `all-MiniLM-L6-v2`.
- **Similarity Computation:**  
  - Cosine similarity is computed between the query and each transcript segment combined with the frame caption.
- **Threshold:**  
  - A threshold cosine similarity score of **0.5** is used to determine whether a valid search result exists.

## Future Enhancements
- **Chat Interaction:**  
  - Use Django Channels to create a WebSocket consumer that handles chat messages.
- **Production Deployment:**  
  - Deploy on AWS.
  - Use MongoDB to store videos.
  - Use Elasticsearch as the vector store for semantic search.

To build and run the project, execute:

```bash
docker compose up --build  
```
   
- The backend will be accessible on http://localhost:8000
- The frontend will be accessible on http://localhost:3000

