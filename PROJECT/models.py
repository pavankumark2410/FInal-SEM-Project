from django.db import models

# Create your models here.
from django.db import models  
from datetime import datetime

class Patient(models.Model):
    p_id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length = 200, default="",)
    email = models.CharField(max_length = 200, default="",)
    phone = models.BigIntegerField()
    date = models.DateTimeField(default=datetime.now())
    def __str__(self):  
        return str(self.name)

class Denoise(models.Model):  
    denoise_id=models.IntegerField(primary_key=True)
    caption = models.CharField(max_length = 200, default="",) 
    image = models.ImageField(upload_to='images')  
    image_output=models.ImageField(upload_to='ouput',default="") 
    uploaded_by=models.IntegerField(default=1)
    uploaded_date=models.DateTimeField(default=datetime.now())
    Patient_name=models.ForeignKey(Patient,default="",on_delete=models.CASCADE)
    def __str__(self):  
        return str(self.caption )


class Radiologist(models.Model):
    r_id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length = 200, default="",)
    password = models.CharField(max_length = 200, default="",)
    newpassword=models.CharField(max_length=200,default="",)
    email = models.EmailField(max_length = 200, default="",)
    phone = models.BigIntegerField()
    date = models.DateTimeField(default=datetime.now())
    def __str__(self):  
        return str(self.name)

class Classification_Report(models.Model):
    c_id = models.IntegerField(primary_key=True)
    caption = models.CharField(max_length = 200, default="",)
    image = models.ImageField(upload_to='classification')  
    tumor_type=models.CharField(max_length = 200, default="",)
    uploaded_by=models.IntegerField(default=1)
    uploaded_date=models.DateTimeField(default=datetime.now())
    Patient_name=models.ForeignKey(Patient,default="",on_delete=models.CASCADE)
    def __str__(self):  
        return str(self.caption)

class Writing(models.Model):
  author = models.IntegerField(primary_key=True)
  subject = models.CharField(max_length=200)
  content = models.TextField()
  create_date = models.DateTimeField()
  modify_date = models.DateTimeField(null=True, blank=True)
  view_count = models.IntegerField(default=0)

  def __str__(self):
    return self.subject

class WritingImage(models.Model):
  writing = models.IntegerField(primary_key=True)
  image = models.ImageField(upload_to='fusion', blank=True, null=True)

