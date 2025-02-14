from django.urls import path
from .views import VideoUploadView, VideoSearchView

urlpatterns = [
    path('upload/', VideoUploadView.as_view(), name='video-upload'),
    path('search/', VideoSearchView.as_view(), name='video-search'),
]