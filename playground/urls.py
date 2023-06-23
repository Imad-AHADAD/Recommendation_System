from django.urls import path
from .views import index, index1, index2

urlpatterns = [path('', index, name="form-index"),
               path('high/<int:predict>/', index1, name="risk-high-index"),
               path('low/<int:predict>/', index2, name="risk-low-index")
               ]
