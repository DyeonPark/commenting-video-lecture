from upload.models import Document
from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader


def execute_commentor(request):
    # 폼을 제출했을 경우 (POST) 그 외 (GET)
    if request.method == "POST":
        # Fetching the form data
        fileTitle = request.POST["fileTitle"]
        videoFile = request.FILES["videoFile"]
        docFile = request.FILES["docFile"]

        document = Document.objects.create(
            title=fileTitle,
            videoFile=videoFile,
            docFile=docFile,
        )
    return HttpResponse("테스트")
