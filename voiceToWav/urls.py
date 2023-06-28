from django.urls import path
from . import views

urlpatterns = [
    path('', views.mainProcess, name='index'),
]
