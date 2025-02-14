from django.db import models

class Video(models.Model):
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to='videos/', blank=True, null=True)
    transcript = models.TextField(blank=True, null=True)  # Full transcript
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

class TranscriptSegment(models.Model):
    video = models.ForeignKey(Video, related_name="segments", on_delete=models.CASCADE)
    start_time = models.FloatField(help_text="Start time in seconds")
    end_time = models.FloatField(help_text="End time in seconds")
    text = models.TextField()

    def __str__(self):
        return f"{self.video.title} [{self.start_time}-{self.end_time}]"
    
class VideoFrame(models.Model):
    video = models.ForeignKey(Video, related_name="frames", on_delete=models.CASCADE)
    timestamp = models.FloatField(help_text="Frame timestamp in seconds")
    caption = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"{self.video.title} - {self.timestamp:.2f} sec"