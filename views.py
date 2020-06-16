from django.shortcuts import render, redirect
from .predictors import *
from django.core.files.storage import FileSystemStorage
import cv2

def home(request):
    return render(request, 'home.html')


def diagnose(request):
    print("hi")
    context = {}
    if request.method == 'POST':
        myfile = request.FILES['myfile']
        taung = request.FILES['taung']
        med = int(request.POST['med'])
        time = int(request.POST['time'])
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        filenamet = fs.save(taung.name, taung)
        file_url = '/Users/aek/PycharmProjects/SeniorProjectWeb/media/' + filename
        file_url = cv2.imread(file_url)
        filet_url = '/Users/aek/PycharmProjects/SeniorProjectWeb/media/' + filenamet
        filet_url = cv2.imread(filet_url)
        arr = (faceinput(file_url))
        eye = predicting_eye([arr[0]])
        mouth = predicting_eye([arr[1]])
        body = findrash(filet_url)
        result = projectprediction([[eye, mouth, body, med, time]])
        if result == "Yes":
            context['url'] = result
            return render(request, "case1.html", context)
        else:
            context['url'] = result
            return render(request, "case2.html", context)

    return render(request, "diagnose.html", context)


def contact(request):
    return render(request, 'contact.html')

def landing(request):
    return redirect('/diagnose/')

# def predicted(request):
#     context = {}
#     x = request.GET.get('input')
#     if x:
#         y = predicting(x)
#         context['answer'] = y
#     return render(request, 'diagnose.html', context)
