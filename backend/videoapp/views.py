from django.shortcuts import render
import os
import requests
from rest_framework import generics, status, views
from rest_framework.response import Response
import tempfile
import logging
import whisper
import traceback
import math
import numpy as np
import cv2
from PIL import Image
import torch

from videoapp.models import Video, TranscriptSegment, VideoFrame
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Load the text embedding model for transcript and caption search.
text_model = SentenceTransformer('all-MiniLM-L6-v2')

# Set up BLIP for frame captioning
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
except Exception as e:
    logger.error("Failed to load BLIP model: " + str(e))
    blip_processor, blip_model = None, None

def extract_frames(video_path, interval_sec=5):
    """
    Extracts frames from the video every interval_sec seconds.
    Returns a list of (timestamp, frame) tuples, where frame is a NumPy array.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_interval = int(fps * interval_sec)
    current_frame = 0
    while current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break
        # Convert from BGR (OpenCV default) to RGB.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp = current_frame / fps
        frames.append((timestamp, frame))
        current_frame += frame_interval
    cap.release()
    return frames

def generate_caption_for_frame(frame):
    """
    Generates a caption for the given frame using BLIP.
    Returns the generated caption as a string.
    """
    if blip_processor is None or blip_model is None:
        return "No caption available."
    image = Image.fromarray(frame.astype("uint8"), "RGB")
    inputs = blip_processor(images=image, return_tensors="pt")
    output_ids = blip_model.generate(**inputs, max_length=16, num_beams=4)
    caption = blip_processor.decode(output_ids[0], skip_special_tokens=True)
    return caption

class VideoUploadView(views.APIView):
    """
    Accepts video uploads, transcribes them using OpenAI Whisper,
    extracts transcript segments and frame captions using BLIP,
    and stores all data in the database.
    """
    def post(self, request):
        video_file = request.FILES.get("video_file")
        title = request.data.get("title", "Untitled Video")
        if not video_file:
            return Response({"error": "No video file provided."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            # Save the uploaded video to a temporary file.
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

            # Create the Video object.
            video_obj = Video.objects.create(title=title, transcript=transcript_text)
            
            # Save transcript segments.
            for seg in segments:
                raw_start = seg.get("start", 0.0)
                raw_end = seg.get("end", 0.0)
                try:
                    start_time = float(raw_start)
                except Exception as ex:
                    logger.error(f"Error converting start time {raw_start}: {ex}")
                    start_time = 0.0
                try:
                    end_time = float(raw_end)
                except Exception as ex:
                    logger.error(f"Error converting end time {raw_end}: {ex}")
                    end_time = 0.0
                if math.isnan(start_time):
                    start_time = 0.0
                if math.isnan(end_time):
                    end_time = 0.0
                TranscriptSegment.objects.create(
                    video=video_obj,
                    start_time=start_time,
                    end_time=end_time,
                    text=seg.get("text", "")
                )
            
            # Extract frames and generate captions.
            frames = extract_frames(tmp_path, interval_sec=5)
            if blip_processor is not None and blip_model is not None:
                for timestamp, frame in frames:
                    caption = generate_caption_for_frame(frame)
                    VideoFrame.objects.create(
                        video=video_obj,
                        timestamp=timestamp,
                        caption=caption
                    )
            else:
                logger.warning("BLIP model not loaded; skipping frame caption generation.")
            
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
    Accepts a natural language query and a video_id, searches across transcript segments
    and video frame captions using Sentence Transformer embeddings,
    and returns the best matching result if above a threshold.
    For transcript segments, returns the start_time; for frame captions, returns the frame timestamp.
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
            return Response({"error": "Video not found."},
                            status=status.HTTP_404_NOT_FOUND)
        
        # Retrieve transcript segments.
        segments_qs = TranscriptSegment.objects.filter(video=video_obj)
        transcript_segments = list(segments_qs.values("start_time", "end_time", "text"))
        texts_transcript = [seg["text"] for seg in transcript_segments]
        
        # Retrieve frame captions.
        frames_qs = VideoFrame.objects.filter(video=video_obj)
        frame_data = list(frames_qs.values("timestamp", "caption"))
        texts_frames = [frame["caption"] for frame in frame_data if frame["caption"]]
        
        # Combine transcript segments and frame captions.
        all_texts = texts_transcript + texts_frames
        results_origin = (["transcript"] * len(texts_transcript)) + (["frame"] * len(texts_frames))
        if not all_texts:
            return Response({"error": "No text available for search."},
                            status=status.HTTP_404_NOT_FOUND)
        
        # Generate embeddings for the combined texts.
        all_embeddings = text_model.encode(all_texts, convert_to_tensor=True)
        query_embedding = text_model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, all_embeddings)[0]
        sorted_indices = np.argsort(-cos_scores.cpu().numpy())
        best_idx = int(sorted_indices[0])
        best_score = float(cos_scores[best_idx])
        
        THRESHOLD = 0.5
        if best_score < THRESHOLD:
            return Response({"result": None, "message": "No answer found."},
                            status=status.HTTP_200_OK)
        
        best_text = all_texts[best_idx]
        best_origin = results_origin[best_idx]
        
        # Determine the timestamp based on the origin.
        timestamp = None
        if best_origin == "transcript":
            for seg in transcript_segments:
                if seg["text"].strip().lower() == best_text.strip().lower():
                    timestamp = seg["start_time"]
                    break
            if timestamp is None and transcript_segments:
                timestamp = transcript_segments[0]["start_time"]
        else:  # best_origin == "frame"
            for frame in frame_data:
                if frame["caption"].strip().lower() == best_text.strip().lower():
                    timestamp = frame["timestamp"]
                    break
            if timestamp is None and frame_data:
                timestamp = frame_data[0]["timestamp"]
        
        best_result = {
            "origin": best_origin,
            "timestamp": timestamp,
            "text": best_text,
            "score": best_score
        }
        return Response({"result": best_result}, status=status.HTTP_200_OK)
