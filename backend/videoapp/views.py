from django.shortcuts import render
import os
import requests
from rest_framework import generics, status, views
from rest_framework.response import Response
import tempfile
import logging
import whisper
import traceback
from videoapp.models import Video, TranscriptSegment
from sentence_transformers import SentenceTransformer, util
import numpy as np

logger = logging.getLogger(__name__)

text_model = SentenceTransformer('all-MiniLM-L6-v2')

class VideoUploadView(views.APIView):
    """
    Accepts video uploads, sends them to Mixpeek for indexing, and returns the index details.
    """
    def post(self, request):
        video_file = request.FILES.get("video_file")
        title = request.data.get("title", "Untitled Video")
        if not video_file:
            return Response({"error": "No video file provided."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                for chunk in video_file.chunks():
                    tmp.write(chunk)
                tmp_path = tmp.name

            logger.info(f"Temporary video file saved to: {tmp_path}")

            try:
                model = whisper.load_model("turbo")
            except Exception:
                model = whisper.load_model("base")
            
            result = model.transcribe(tmp_path)
            transcript_text = result.get("text", "")
            segments = result.get("segments", [])

            logger.info("Transcription complete.")

            video_obj = Video.objects.create(title=title, transcript=transcript_text)
            
            for seg in segments:
                TranscriptSegment.objects.create(
                    video=video_obj,
                    start_time=seg.get("start", 0.0),
                    end_time=seg.get("end", 0.0),
                    text=seg.get("text", "")
                )
            
            return Response({
                "video_id": video_obj.id,
                "transcript": transcript_text,
                "segments": segments
            }, status=status.HTTP_200_OK)
        except Exception as e:
            error_msg = traceback.format_exc()
            logger.error("Error during transcription:\n" + error_msg)
            return Response({
                "error": "Failed to process video.",
                "details": error_msg
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class VideoSearchView(views.APIView):
    """
    Accepts a natural language query (and a video ID if needed) and returns search results.
    """
    def post(self, request):
        query = request.data.get("query")
        video_id = request.data.get("video_id")
        if not query or not video_id:
            return Response({"error": "Missing query or video_id."},
                            status=status.HTTP_400_BAD_REQUEST)
        
        try:
            video_obj = Video.objects.get(id=video_id)
        except Video.DoesNotExist:
            return Response({"error": "Video not found."}, status=status.HTTP_404_NOT_FOUND)
        
        segments_qs = TranscriptSegment.objects.filter(video=video_obj)
        if not segments_qs.exists():
            return Response({"error": "No transcript segments found."}, status=status.HTTP_404_NOT_FOUND)
        
        segments = list(segments_qs.values("start_time", "end_time", "text"))
        texts = [seg["text"] for seg in segments]

        # Generate embeddings for the transcript segments
        segment_embeddings = text_model.encode(texts, convert_to_tensor=True)
        query_embedding = text_model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, segment_embeddings)[0]
        top_results = np.argpartition(-cos_scores.cpu().numpy(), range(3))[0:3]

        results = []
        for idx in top_results:
            results.append({
                "start": segments[idx]["start_time"],
                "end": segments[idx]["end_time"],
                "text": segments[idx]["text"],
                "score": float(cos_scores[idx])
            })
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        THRESHOLD = 0.5
        if results[0]["score"] < THRESHOLD:
            return Response({"result": None, "message": "No answer found."},status=status.HTTP_200_OK)
        else:
            logger.info(f"Search results for query '{query}' on video '{video_obj.title}': {results}")
            return Response({"result": results[0]}, status=status.HTTP_200_OK)
            
