from django.forms import ModelForm

from inpainting.models import Image


class ImageCreationForm(ModelForm):
    class Meta:
        model = Image
        fields = ['image']