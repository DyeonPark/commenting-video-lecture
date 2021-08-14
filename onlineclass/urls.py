from . import views
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

app_name = "onlineclass"

urlpatterns = [
    path("commentor/", views.execute_commentor, name="execute_commentor"),
]