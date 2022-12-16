from django.shortcuts import render, redirect
from .forms import SignUpForm, LoginForm
from django.contrib.auth import authenticate, login
from PROJECT.models import *
# Create your views here.


def index(request):
    return render(request, 'index.html')


def register(request):
    msg = None
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            msg = 'user created'
            return redirect('login_view')
        else:
            msg = 'form is not valid'
    else:
        form = SignUpForm()
    return render(request,'registration/register.html', {'form': form, 'msg': msg})


def login_view(request):
    form = LoginForm(request.POST or None)
    msg = None
    if request.method == 'POST':
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None and user.is_admin:
                login(request, user)
                return redirect('adminpage')
            elif user is not None and user.is_customer:
                login(request, user)
                return redirect('patient')
            elif user is not None and user.is_employee:
                login(request, user)
                return redirect('radiologist')
            else:
                msg= 'invalid credentials'
        else:
            msg = 'error validating form'
    return render(request, 'registration/login.html', {'form': form, 'msg': msg})


def admin(request):
    return redirect("\admin")


def patient(request):
    
    return render(request,'index.html',{"name": request.user.username})


def radiologist(request):
    Denoisecount=Denoise.objects.all().count()
    Classification_Report_Count=Classification_Report.objects.all().count()
    Writing_Count=Writing.objects.all().count()
    total=Denoisecount+Classification_Report_Count+Writing_Count
    return render(request,'home/index.html',{"name": request.user.username,"Denoisecount":Denoisecount,"Writing_Count":Writing_Count,"Classification_Report_Count":Classification_Report_Count,"total":total})

def reports(request):
    return render(request,'reports/reports.html')