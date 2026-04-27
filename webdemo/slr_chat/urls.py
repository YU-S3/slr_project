from django.urls import path
from . import views

app_name = 'slr_chat'

urlpatterns = [
    path('', views.index, name='index'),
    path('settings/', views.settings_view, name='settings'),
]