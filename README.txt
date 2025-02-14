This projectallows users to upload a video, transcribe its content using OpenAI’s Whisper, and perform semantic searches over the transcript using a Sentence Transformer model. I used Python Django for the backend and Next.js for the frontend. Containerization is achieved using Docker, and Poetry is used for managing Python dependencies.

Features
Video Upload & Transcription: The backend accepts video uploads and uses OpenAI’s Whisper to generate a transcription and timestamped segments.
Semantic Search: The transcript is encoded into embeddings using the SentenceTransformer model all-MiniLM-L6-v2.
Cosine similarity is computed between the query and each transcript segment.
A threshold cosine similarity score of 0.5 is used to determine whether a valid search result exists.

Future Enhancements:

Integrate OpenCV for frame extraction combined with an image captioning model such as BLIP to capture visual content and generate additional embeddings.
For the bonus questions, I would Django Channels to create a WebSocket consumer that handles chat messages.
For production deployment, the plan is to deploy on AWS and use mongodb to store the videos and use Elasticsearch for vector store and semantic search, with a potential upgrade to use Azure Video Indexer if the budget allows.

Running project with Docker Compose:
docker compose up --build
The backend should be accessible on http://localhost:8000 and the frontend on http://localhost:3000.

