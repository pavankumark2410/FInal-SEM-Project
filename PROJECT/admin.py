from django.contrib import admin
from .models import *
# Register your models here.
admin.site.register(Denoise)
admin.site.register(Radiologist)
admin.site.register(Patient)
admin.site.register(Classification_Report)
admin.site.register(WritingImage)
admin.site.register(Writing)