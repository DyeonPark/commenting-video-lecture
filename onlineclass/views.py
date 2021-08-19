from upload.models import Document
from onlineclass.models import Helper
from django.http import HttpResponse
from django.template import loader


def execute_commentor(request, doc_id):
    template = loader.get_template('onlineclass/commentor.html')

    doc = Document.objects.get(id=doc_id)
    helper = Helper.objects.get(doc_id=doc_id)

    context = {
        "doc": doc,
        "helper": helper,
    }

    return HttpResponse(template.render(context, request))
