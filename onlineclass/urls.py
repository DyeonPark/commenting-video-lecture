from . import views
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

app_name = "onlineclass"

urlpatterns = [
    path("commentor/<int:doc_id>/", views.execute_commentor, name="execute_commentor"),
    path("pdf_download/<int:doc_id>/", views.download_pdf, name="download_pdf"),
]