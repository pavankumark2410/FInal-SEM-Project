# forms.py
from django import forms
from .models import *
from django.forms import ClearableFileInput
class Denoiseform(forms.ModelForm):
  
    class Meta:
        model = Denoise
        fields = [ 'image','Patient_name']

class Radiologistform(forms.ModelForm):
    class Meta:
        model = Radiologist
        fields =['name','email','password','newpassword','phone']
    
    def save(self, commit=True):
        user = super(Radiologistform, self).save(commit=False)
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
        return user

class Patientform(forms.ModelForm):
    class Meta:
        model=Patient
        fields=['name','email','phone']

class classificationform(forms.ModelForm):
    class Meta:
        model=Classification_Report
        fields=[ 'image','Patient_name']

class WritingForm(forms.ModelForm):
  class Meta:
    model = Writing
    fields = ['subject', 'content']

    labels = {
      'subject': 'SUBJECT',
      'content': 'CONTENT',

    }

class WritingFullForm(WritingForm):
  images = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True}))

  class Meta(WritingForm.Meta):
    fields = WritingForm.Meta.fields + ['images', ]


class AuthenticationForm(forms.ModelForm):
    class Meta:
        model = Radiologist
        fields=['email','password']