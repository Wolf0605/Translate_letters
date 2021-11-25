from django.shortcuts import render

# Create your views here.
from django.urls import reverse
from django.views.generic import CreateView, DetailView

from inpainting.forms import ImageCreationForm
from inpainting.models import Image


class ImageCreationView(CreateView):
    model = Image
    form_class = ImageCreationForm
    template_name = 'inpainting/create.html'

    def get_success_url(self):
        return reverse('inpaintingapp:result', kwargs={'pk':self.object.pk})

class ImageDetailView(DetailView):
    model = Image
    context_object_name = 'target_image'
    template_name = 'inpainting/result.html'