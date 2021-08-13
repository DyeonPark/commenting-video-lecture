from django.db import models

# Create your models here.
class Document(models.Model):
	title = models.CharField(max_length = 200)
	videoFile = models.FileField(upload_to = "Uploaded Files/")
	docFile = models.FileField(upload_to = "Uploaded Files/")
	dateTiemOfUpload = models.DateTimeField(auto_now = True)
