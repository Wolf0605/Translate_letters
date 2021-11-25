from django.urls import path

from inpainting.views import ImageCreationView, ImageDetailView

app_name = 'inpaintingapp'
urlpatterns = [
    path('',ImageCreationView.as_view(), name = 'create'),
    path('result/<int:pk>', ImageDetailView.as_view(), name='result'),

]