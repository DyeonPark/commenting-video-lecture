from django.db import models
from upload.models import Document


# 최종 Commentor 서비스를 위한 모델
class Helper(models.Model):
	helper_id = models.AutoField(primary_key=True)
	doc_id = models.ForeignKey(Document, on_delete=models.CASCADE)
	helper_audio = models.FileField()
	# helper_txt = models.FileField()
	helper_csv = models.FileField(null=True)
