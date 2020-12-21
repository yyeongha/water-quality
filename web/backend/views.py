import json
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_protect
from django.conf import settings


def index(request):
    context = {}
    return render(request, 'backend/index.html', context)


def sample_map(request):
    context = {}
    return render(request, 'backend/sample_map.html', context)


def sample_datepicker(request):
    context = {}
    return render(request, 'backend/sample_datepicker.html', context)
    