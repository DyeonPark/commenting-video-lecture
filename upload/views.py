from . import models
from django.shortcuts import render

def uploadFile(request):
    if request.method == "POST":
        # Fetching the form data
        fileTitle = request.POST["fileTitle"]
        videoFile = request.FILES["videoFile"]
        docFile = request.FILES["docFile"]

        # Saving the information in the database
        document = models.Document(
            title = fileTitle,
            videoFile = videoFile,
            docFile = docFile
        )
        document.save()

    documents = models.Document.objects.all()

    return render(request, "upload/upload_file.html", context = {
        "files": documents
    })