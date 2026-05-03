from django.shortcuts import render

def index(request):
    return render(request, 'slr_chat/index.html')

def settings_view(request):
    return render(request, 'slr_chat/settings.html')
