from django.urls import path
from . import views
from django.conf import settings   
from django.conf.urls.static import static  
urlpatterns = [
    path('index', views.index),
    path('image_upload',views.image_upload, name="image_upload"),
    path('noise_removal', views.noise_removal, name = 'noise_removal'),
    path('success', views.success, name = 'success'),
    path('', views.homepage, name = 'homepage'),
    path('dashboard', views.dashboard, name = 'dashboard'),
    #path('login', views.login, name = 'login'),
    #path('register', views.register, name = 'register'),
    path('classification',views.classification,name="classification"),
    path('classification_image_upload',views.classification_image_upload,name="classification_image_upload"),
    path('fusion_ct_mri',views.fusion_ct_mri,name="fusion_ct_mri"),
    path("fusion",views.writing_create,name="fusion"),
    path("filters_noise",views.filters_noise,name="filters_noise"),
    path("colouring",views.colouring,name="colouring"),
    path("detection_tumor",views.detection_tumor,name="detection_tumor"),
    path("classify_fusion",views.classify_fusion,name="classify_fusion"),
]
if settings.DEBUG:  
        urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)  