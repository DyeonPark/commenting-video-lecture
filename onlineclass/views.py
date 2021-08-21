from upload.models import Document
from onlineclass.models import Helper

from django.http import HttpResponse
from django.template import loader

from django.http import FileResponse
from django.core.files.storage import FileSystemStorage


def download_pdf(request, doc_id):
    doc = Document.objects.get(id=doc_id)

    file_path = doc.docFile.path
    fs = FileSystemStorage(file_path)

    set_filename = doc.title + " 강의자료.pdf"  # 파일명 지정
    print(set_filename)

    response = FileResponse(fs.open(file_path, 'rb'))
    response['Content-Disposition'] = 'attachment; filename=' + set_filename  # !!! 파일명 지정이 안되는 오류 발생 !!!
    return response


def execute_commentor(request, doc_id):
    template = loader.get_template('onlineclass/commentor.html')

    doc = Document.objects.get(id=doc_id)
    helper = Helper.objects.get(doc_id=doc_id)

    context = {
        "doc": doc,
        "helper": helper,
    }

    return HttpResponse(template.render(context, request))

def download_txt(request, doc_id):
    doc = Document.objects.get(id=doc_id)
    help = Helper.objects.get(doc_id=doc_id)

    file_path = help.helper_txt.path
    fs = FileSystemStorage(file_path)

    set_filename = doc.title + " 강의자료 해설파일.txt"  # 파일명 지정
    print(set_filename)

    response = FileResponse(fs.open(file_path, 'rb'))
    response['Content-Disposition'] = 'attachment; filename=' + set_filename  # !!! 파일명 지정이 안되는 오류 발생 !!!
    return response
