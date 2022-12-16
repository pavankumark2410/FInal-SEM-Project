from django.urls import path
from . import views

urlpatterns = [
    path('index/', views.index, name= 'index'),
    path('login/', views.login_view, name='login_view'),
    path('register/', views.register, name='register'),
    path('adminpage/', views.admin, name='adminpage'),
    path('patient/', views.patient, name='patient'),
    path('radiologist/', views.radiologist, name='radiologist'),
    path('reports/', views.reports, name='reports'),
]