from django.urls import path
from . import views

urlpatterns=[
    path('',views.predictor,name='predictor'),
    path('form',views.check,name='check'),
    path('result',views.forminfo,name='result'),
]
