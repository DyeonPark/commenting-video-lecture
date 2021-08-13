import os
import shutil
import datetime
from . import models
from upload.models import Document
from django.shortcuts import render

def uploadFile(request):
    if request.method == "POST":

        # Fetching the form data
        fileTitle = request.POST["fileTitle"]
        videoFile = request.FILES["videoFile"]
        docFile = request.FILES["docFile"]

        new_path = getTime()

        # Saving the information in the database
        document = models.Document(
            title = fileTitle,
            videoFile = videoFile,
            docFile = docFile,
            dateTiemOfUpload = new_path
        )
        document.save()

        # 사용자가 업로드한 파일 목록 불러오기
        original_path = "D:/commentor/media/Uploaded_Files/"
        files = os.listdir(original_path)
        
        # 디렉토리 생성
        new_path = "D:/commentor/media/" + new_path + "/"        
        createFolder(new_path)

        # 파일 이동
        for file in files:
            shutil.move(original_path + file, new_path + file)

        # 모델 내 path(url) 수정
        obj = Document.objects.get(title = fileTitle)
        obj.videoFile = new_path + "lecture_video.mp4"
        obj.docFile = new_path + "lecture_doc.pdf"
        obj.save()

    documents = models.Document.objects.all()

    return render(request, "upload/upload_file.html", context = {
        "files": documents
    })

def createFolder(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print ('Error: Creating directory. ' +  dir)

def getTime():
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d %H-%M-%S')
    return nowDatetime