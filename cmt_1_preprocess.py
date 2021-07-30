# """
# * 코드 파일 이름: cmt_1_preprocess.py
# * 코드 작성자: 박동연, 강소정, 김유진
# * 코드 설명: 동영상 강의 해설 파일을 생성하기 위한 전처리 파일 생성
# * 코드 최종 수정일: 2021/07/30 (박동연)
# * 문의 메일: yeon0729@sookmyung.ac.kr
# """

# 패키지 및 라이브러리 호출
import cv2
from skimage.measure import compare_ssim

from PIL import Image
import sys
import time
import datetime
import pandas as pd
import re
import pdfplumber
from pytesseract import *

import shutil
import moviepy.editor as mp

from pdf2image import convert_from_path  # pdf2img
from gtts import gTTS
from pydub import AudioSegment

# 이미지 추출 import
import fitz  # PyMuPDF
import io
import os

# 이미지 캡션 import
import requests

# 번역 import
from googletrans import Translator

pdf2image_module_path = "data/Release-21.03.0/poppler-21.03.0/Library/bin/"

# 강의 동영상 내 전환시점 파악을 위한 라이브러리 호출
from scenedetect import VideoManager, SceneManager, StatsManager
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import save_images, write_scene_list_html


# 경로 설정 (경로 내에 한글 디렉토리 및 한글 파일이 있으면 제대로 동작하지 않음 유의 !!!!!)
default_path = "UIUX/"
pdf_path = default_path + "lecture_doc.pdf"
video_path = default_path + "lecture_video.mp4"
audio_path = default_path + "lecture_audio.mp3"
capture_path = default_path + "capture/"
capture_FA_path = default_path + "capture_FA/"
slide_path = default_path + "slide/"
txt_path = default_path + "txt/"
tts_path = default_path + "tts/"
mix_path = default_path + "mix/"
lec_path = default_path + "lec/"
img_path = default_path + "img/"

# 최종 출력 파일
df = pd.DataFrame()
save_path = default_path + "transform_timeline_result.csv"


# 의도한바와 같이 정렬될 수 있도록 파일번호 수정하여 반환하는 함수 (최대 9999장까지 가능)
def set_Filenum_of_Name(filenum):
    fileName = ""

    if (filenum < 10):  # 파일번호가 한자리일때
        fileName = "000" + str(filenum)
    elif (filenum >= 10 and filenum < 100):  # 파일번호가 두자리일때
        fileName = "00" + str(filenum)
    elif (filenum >= 100 and filenum < 1000):  # 파일번호가 세자리일때
        fileName = "0" + str(filenum)
    elif (filenum >= 1000 and filenum < 10000):  # 파일번호가 네자리일때
        fileName = str(filenum)
    else:
        sys.exit(">>> 파일이 너무 큽니다 - 9999장 이상")

    return fileName


# pdf 파일을 jpg 파일로 변환하는 함수
def pdf2jpg():  # pdf 파일을 기반으로 이미지(jpg)를 생성하는 함수

    print("\n[PDF2IMG 시작] PDF2JPG 이미지 변환을 시작합니다")

    pages = convert_from_path(pdf_path, poppler_path=pdf2image_module_path)
    print(">>> 인식된 강의자료 페이지 수:", len(pages))

    # 디렉토리 유무 검사 및 디렉토리 생성
    try:
        if not os.path.exists(slide_path):  # 디렉토리 없을 시 생성
            os.makedirs(slide_path)
    except OSError:
        print('Error: Creating directory. ' + txt_path)  # 디렉토리 생성 오류

    for i, page in enumerate(pages):
        page.save(slide_path + "slide_" + set_Filenum_of_Name(i + 1) + ".jpg", "JPEG")

    print("[PDF2IMG 종료] PDF2JPG 이미지 변환을 종료합니다\n")


# 동영상 내 화면 전환이 발생하는 시점을 기준으로 동영상 캡처(추출)을 하는 함수
def capture_video():
    print("\n[전환장면 캡처 시작] 영상 내 전환 시점을 기준으로 이미지 추출을 시작합니다")

    video_manager = VideoManager([video_path])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)

    # 가장 예민하게 잡아내도록 1~100 중 1로 설정
    scene_manager.add_detector(ContentDetector(threshold=1))

    video_manager.set_downscale_factor()

    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    scene_list = scene_manager.get_scene_list()
    print(">>>", f'{len(scene_list)} scenes detected!')  # 전환 인식이 된 장면의 수

    save_images(
        scene_list,
        video_manager,
        num_images=1,
        image_name_template='$SCENE_NUMBER',
        output_dir=capture_path)

    write_scene_list_html(default_path + 'SceneDetectResult.html', scene_list)

    captured_timeline_list = []  # 전환된 시점을 저장할 리스트 함수
    for scene in scene_list:
        start, end = scene

        # 전환 시점 저장
        # print(f'{start.get_seconds()} - {end.get_seconds()}')
        captured_timeline_list.append(start.get_seconds())

    print("[전환장면 캡처 종료] 영상 내 전환 시점을 기준으로 이미지 추출을 종료합니다\n")
    return captured_timeline_list


# pdf 파일에 있는 텍스트를 슬라이드 별로 뽑아내는 함수
def pdf2txt():
    print("\n[PDF to TXT 변환 시작] 슬라이드 이미지를 텍스트로 변환을 시작합니다")

    # 디렉토리 유무 검사 및 디렉토리 생성
    try:
        if not os.path.exists(txt_path):  # 디렉토리 없을 시 생성
            os.makedirs(txt_path)
    except OSError:
        print('Error: Creating directory. ' + txt_path)  # 디렉토리 생성 오류

    textt = ""
    textt1 = ""
    txt_res = ""
    table_final_text = []
    text_com = ""
    table_list = []
    a = 1
    b = 1

    Pdf = pdfplumber.open(pdf_path)

    for page_idx, page in enumerate(Pdf.pages):
        txtFile = open(txt_path + set_Filenum_of_Name(page_idx + 1) + ".txt", "w", -1, "utf-8")  # 번역한 내용을 저장할 텍스트 파일

        txtFile.write(str(page_idx + 1) + "번 슬라이드 해설 시작" + "\n" + "\n")

        # 텍스트->table
        result = page.extract_text()
        text = str(page.extract_text())
        # text = text.replace('\n'," ")
        text = re.sub('\\n+', '\n', text)
        text = text + "\n"

        for table in page.extract_tables():
            for row in table:
                for column in range(0, len(row)):
                    text_com = text_com + row[column] + " "
                    textt = str(a) + "행"
                    textt1 = " " + str(b) + "열 " + row[column] + "\n"
                    txt_res = txt_res + textt + textt1

                    b = b + 1
                b = 1
                a = a + 1
                text_com = text_com[:-1]
                text_com = text_com + "\n"
            table_new = '표 시작\n' + txt_res + '표 끝 \n'
            table_final_text.append(table_new)
            table_list.append(text_com)
            # print(text_com)
            txt_res = ""
            text_com = ""
            a = 1
            b = 1

        # 공백 O
        for i, j in zip(table_list, table_final_text):
            text = text.replace(i, j)

        imgcaption = imgExtract(page_idx, text)

        if (imgcaption == "이미지 없음"):
            print("이미지 없음")
            txtFile.write(text + "\n")
        else:
            imgcaption = "".join(imgcaption)
            txtFile.write(text + imgcaption + "\n")

        txtFile.close()

        # 텍스트 변환 필터링
        NLP(txt_path + set_Filenum_of_Name(page_idx + 1) + ".txt")

        # 이미지 캡션 위치 조정
        modifytxt(txt_path + set_Filenum_of_Name(page_idx + 1) + ".txt", page_idx)

        print(">>>", page_idx + 1, "번째 PDF 슬라이드 텍스트 변환 완료")

    Pdf.close()
    print("[PDF to TXT 변환 종료] 슬라이드 이미지를 텍스트로 변환을 종료합니다\n")


# 텍스트 변환 필터링 함수
def NLP(filename):
    txtfilter_open = open(filename, "r", -1, "utf-8")
    pp = re.compile("[ㄱ-ㅣ가-힣A-Za-z0-9-+.()~\s]")
    txtfilter1 = txtfilter_open.read()
    txtfilter1 = pp.findall(txtfilter1)
    txtfilter1 = ''.join(txtfilter1)
    txtfilter2 = re.sub('\\n+', '\n', txtfilter1)
    textfilter = re.sub('', '', txtfilter2)
    txtfilter_out = open(filename, "w", -1, "utf-8")
    txtfilter_out.write(textfilter)
    txtfilter_out.close()


# 이미지 캡션 위치 조정하는 함수
def modifytxt(filename, page_idx):
    result = []
    text = []
    string = '그림'

    with open(filename, "r", -1, "utf-8") as file:
        for line in file:
            if line.startswith(" ") and line[1:].startswith(string):
                line = line[1:]  # 맨 앞 공백 제거
                result.append(line)
                line.replace(line, " ")
            elif line.startswith(string):
                result.append(line)
                line.replace(line, " ")
            else:
                text.append(line)
        result.sort()  # 그림 순서 정렬

    with open(filename, "w", -1, "utf-8") as outfile:
        for j in range(0, len(text)):
            outfile.write(str(text[j]))
        for j in range(0, len(result)):
            outfile.write(str(result[j]))
        outfile.write(str(page_idx + 1) + "번 슬라이드 해설 끝\n")
    outfile.close()


# pdf 파일에 이미지가 있는지 판단하는 함수
def imgExtract(page_index, result):  # 이미지 추출 함수
    # open the file
    pdf_file = fitz.open(pdf_path)

    # 디렉토리 유무 검사 및 디렉토리 생성
    try:
        if not os.path.exists(img_path):  # 디렉토리 없을 시 생성
            os.makedirs(img_path)
    except OSError:
        print('Error: Creating directory. ' + img_path)  # 디렉토리 생성 오류

    # get the page itself
    page = pdf_file[page_index]
    image_list = page.getImageList()
    # printing number of images found in this page
    if image_list:
        print(f"[+] Found a total of {len(image_list)} images in page {page_index}")

        caption_list_eng = []
        caption_list_kor = []

        for image_index, img in enumerate(page.getImageList(), start=1):
            # get the XREF of the image
            xref = img[0]
            # extract the image bytes
            base_image = pdf_file.extractImage(xref)
            image_bytes = base_image["image"]
            # get the image extension
            image_ext = base_image["ext"]
            # load it to PIL
            image = Image.open(io.BytesIO(image_bytes))

            if (image.size[0] <= 50 or image.size[1] <= 50 or image.size[0] >= 10000 or image.size[1] >= 10000):

                if image.size[0] < image.size[1]:
                    new_width = 60
                    new_height = int(new_width * image.size[1] / image.size[0])

                else:
                    new_height = 60
                    new_width = int(new_height * image.size[0] / image.size[1])

                try:
                    resize_image = image.resize((new_width, new_height), Image.ANTIALIAS)
                    imgfileName = "slide_" + set_Filenum_of_Name(page_index + 1) + "img_" + set_Filenum_of_Name(
                        image_index) + ".jpg"

                except OSError:
                    pass
                    resize_image.save(img_path + imgfileName)
                    resize_image = resize_image.convert("RGB")
                    resize_image.save(img_path + imgfileName)

            else:
                try:
                    imgfileName = "slide_" + set_Filenum_of_Name(page_index + 1) + "img_" + set_Filenum_of_Name(
                        image_index) + ".jpg"
                except OSError:
                    pass
                image = image.convert("RGB")
                image.save(open(img_path + imgfileName, "wb"))

            if (result.find('그림 ' + str(image_index)) != -1):
                os.remove(img_path + imgfileName)
                print("그림 " + str(image_index) + " 삭제 완료")

            else:
                image_caption = imgCaption(imgfileName)  # 이미지 캡션 함수 호출
                caption_list_eng.append(image_caption)
                image_caption = "그림 " + str(image_index) + " " + eng2Kor(image_caption) + "\n"
                caption_list_kor.append(image_caption)

        print("이미지캡션 영어:", caption_list_eng)
        print("이미지캡션 한국어:", caption_list_kor)

    else:
        print("[!] No images found on page", page_index)
        caption_list_kor = "이미지 없음"

    return caption_list_kor


# 이미지의 캡션 자막(영어)를 생성하는 함수
def imgCaption(imgfileName):  # 이미지 캡션 함수

    subscription_key = '1f05b01e39814389bff7f16810cc0522'
    endpoint = 'https://commentor.cognitiveservices.azure.com/'

    analyze_url = endpoint + "vision/v3.1/analyze"
    image_data = open(img_path + imgfileName, "rb").read()
    headers = {'Ocp-Apim-Subscription-Key': subscription_key,
               'Content-Type': 'application/octet-stream'}
    params = {'visualFeatures': 'Categories,Description,Color'}
    response = requests.post(
        analyze_url, headers=headers, params=params, data=image_data)
    response.raise_for_status()

    analysis = response.json()

    image_caption = analysis["description"]["captions"][0]["text"].capitalize()

    return image_caption


# 영어 문자열을 한글 문자열로 번역하는 함수
def eng2Kor(image_caption):
    translator = Translator()
    trans1 = translator.translate(image_caption, src='en', dest='ko')
    return trans1.text

# 캡처 이미지 중 최초 등장 시점을 파악하여 필터링하는 함수
def captureFiltering():
    
    print("\n[이미지 유사도 계산 시작] 캡처 화면 필터링을 시작합니다")
    captureList = os.listdir(capture_path)
    captureList = [capture_file for capture_file in captureList if capture_file.endswith(".jpg")]  # jpg로 끝나는 것만 가져오기
    captureList.sort()
    print(">>> 캡쳐 파일 목록:", captureList)

    capture_img_load = []  # 이미지 파일 리스트를 사용하여 캡처 이미지 불러오기
    for i in captureList:
        img = cv2.imread(capture_path + i)  # 캡처 이미지
        capture_img_load.append(img)

    # 유사도 측정
    selected_idx = []
    for idx in range(0, len(capture_img_load) - 1):
        (score, diff) = compare_ssim(capture_img_load[idx], capture_img_load[idx + 1], full=True, multichannel=True)
        print(idx + 1, "vs", idx + 2, "Similarity:", score)

        # 최초 등장 시점의 인덱스 저장
        if idx == 0:  # 가장 첫 번째 슬라이드 최초 등장시점 저장
            selected_idx.append(idx)
        if score <= 0.95:  # 두 번째 슬라이드부터 최초 등장 시점 저장
            selected_idx.append(idx + 1)

        # 디렉토리 유무 검사 및 디렉토리 생성
        try:
            if not os.path.exists(capture_FA_path):  # 디렉토리 없을 시 생성
                os.makedirs(capture_FA_path)
        except OSError:
            print('Error: Creating directory. ' + capture_FA_path)  # 디렉토리 생성 오류

    # 최초 등장 시점의 인덱스에 맞는 이미지 파일을 복사
    for idx in selected_idx:
        shutil.copy2(capture_path + captureList[idx], capture_FA_path + captureList[idx])
        print(">>> >>>", "'" + captureList[idx] + "'", " 파일 복사 완료")

    print("\n[이미지 유사도 계산 시작] 캡처 화면 필터링을 종료합니다")

    return selected_idx


# 동영상 캡처 이미지와 원본 슬라이드 이미지 간 유사도를 계산하는 함수
def orbCompare(capture):
    print("\n[ORB 계산 시작] 캡쳐 화면과 슬라이드 이미지의 ORB 유사도 계산을 시작합니다")

    slideList = os.listdir(slide_path)
    slideList = [slide_file for slide_file in slideList if slide_file.endswith(".jpg")]  # jpg로 끝나는 것만 가져오기
    slideList.sort()
    print(">>> 슬라이드 파일 목록:", slideList)

    orb = cv2.ORB_create()

    capture_img = cv2.imread(capture_FA_path + capture, None)
    kp_c, des_c = orb.detectAndCompute(capture_img, None)

    match_list = []

    for slide in slideList:
        slide_img = cv2.imread(slide_path + slide, None)
        kp_s, des_s = orb.detectAndCompute(slide_img, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        matches = bf.match(des_c, des_s)
        matches = sorted(matches, key=lambda x: x.distance)

        print(">>>", capture, "vs", slide, ":", len(matches))
        match_list.append(len(matches))

    print("*" * 50)
    slide_max_idx = match_list.index(max(match_list))
    result = ">>> " + str(capture) + " - " + str(slideList[slide_max_idx]) + " : " + str(match_list[slide_max_idx])
    print(result)
    print("*" * 50)
    print("[ORB 계산 종료] 캡쳐 화면과 슬라이드 이미지의 ORB 유사도 계산을 종료합니다\n")

    return slide_max_idx


# 텍스트 자카드 유사도 판별 함수
def JaccardSimilarity(inp1, inp2):
    list_inp1 = inp1.split()
    list_inp2 = inp2.split()
    mom = set(list_inp1).union(set(list_inp2))
    son = set(list_inp1).intersection(set(list_inp2))
    #print(mom)
    #print(son)
    #print("\n")
    return len(son)/len(mom)


# 텍스트 유사도 + ORB 유사도를 사용하여 알맞는 슬라이드를 찾아주는 함수
def txtSimCompare():
    slideList = os.listdir(slide_path)
    slideList = [slide_file for slide_file in slideList if slide_file.endswith(".jpg")]  # jpg로 끝나는 것만 가져오기
    slideList.sort()
    print(">>> 슬라이드 파일 목록:", slideList)

    capture_FA_List = os.listdir(capture_FA_path)
    capture_FA_List = [capture_file for capture_file in capture_FA_List if capture_file.endswith(".jpg")]  # jpg로 끝나는 것만 가져오기
    capture_FA_List.sort()
    print(">>> 캡쳐 파일 목록:", capture_FA_List)

    txtFile = open(default_path + "OCRTXT_result.txt", "w", -1, "utf-8")

    match_list = []
    answer = 0
    for capture in capture_FA_List:
        result_list = []
        jpgtotext = pytesseract.image_to_string(Image.open(capture_FA_path + capture), lang='kor+eng')

        Pdf = pdfplumber.open(pdf_path)

        for page in Pdf.pages:

            text = str(page.extract_text())

            result = JaccardSimilarity(jpgtotext, text) # 선 텍스트 유사도 비교
            result_list.append(result)
            print(capture, "vs", str(page), ":", str(result))

        if max(result_list) <= 0.25: # 텍스트 유사도 비교가 부정확할 시 orb 사용
            page_num = orbCompare(capture) + 1
        else:
            page_num = result_list.index(max(result_list)) + 1

        match_list.append(page_num)
        answer = ">>> " + str(capture) + " - Page " + str(page_num) + " : " + str(max(result_list)) + "\n"
        print(answer)
        txtFile.write(answer)

    txtFile.close()

    print("*****", match_list)

    final = []
    for i in match_list:
        final.append("slide_" + set_Filenum_of_Name(i) + ".jpg")

    return final


# 텍스트 파일을 기반으로 TTS 음성파일을 생성하는 함수
def txt2TTS():
    print("\n[TTS 시작] TTS 변환을 시작합니다")
    txt_list = os.listdir(txt_path)
    txt_list = [txt_file for txt_file in txt_list if txt_file.endswith(".txt")]  # jpg로 끝나는 것만 가져오기
    txt_list.sort()
    print(">>> 텍스트 파일 목록:", txt_list)

    # 디렉토리 유무 검사 및 디렉토리 생성
    try:
        if not os.path.exists(tts_path):  # 디렉토리 없을 시 생성
            os.makedirs(tts_path)
    except OSError:
        print('Error: Creating directory. ' + tts_path)  # 디렉토리 생성 오류

    for idx, txt_file in enumerate(txt_list):
        infile = txt_path + txt_file
        f = open(infile, 'r', encoding='UTF-8')
        sText = f.read()
        f.close()

        tts = gTTS(text=sText, lang='ko', slow=False)
        tts.save(tts_path + "tts_" + set_Filenum_of_Name(idx + 1) + ".mp3")

        print(set_Filenum_of_Name(idx + 1) + " MP3 file saved!")

    print("[TTS 종료] TTS 변환을 종료합니다\n")


# 동영상 내 화면 전환이 발생하는 시점을 기준으로 원본 오디오 파일을 자르는 함수
def cutLectureMp3():
    print("\n[lec 생성 시작] mp3 파일 변환 및 mp3 파일 CUT을 시작합니다")

    print(">>> mp4 영상 → mp3 오디오 변환 시작")
    # 원본 강의 mp4영상을 mp3 형식으로 변환
    clip = mp.VideoFileClip(video_path)  # 동영상 불러오기
    clip.audio.write_audiofile("lecture_audio.mp3")  # mp3 파일로 변환
    shutil.move("lecture_audio.mp3", audio_path)  # mp3 파일을 원하는 디렉토리로 이동
    print(">>> mp4 영상 → mp3 오디오 변환 완료")

    print(">>> mp3 오디오 CUT 시작")
    # 원본 강의 mp3 파일을 전환 시간에 맞추어 cut
    audio = AudioSegment.from_mp3(audio_path)
    time_csv = pd.read_csv(save_path)

    # 디렉토리 유무 검사 및 디렉토리 생성
    try:
        if not os.path.exists(lec_path):  # 디렉토리 없을 시 생성
            os.makedirs(lec_path)
    except OSError:
        print('Error: Creating directory. ' + lec_path)  # 디렉토리 생성 오류

    for i in range(len(time_csv["time"])):

        fileName = "lec_" + set_Filenum_of_Name(i + 1) + ".mp3"
        fileName = lec_path + fileName

        if i == (len(time_csv["time"]) - 1):  # 마지막 클립
            result = audio[int(time_csv["time"][i]) * 1000:]
            result.export(fileName, format='mp3')
        else:  # 처음, 중간 클립
            result = audio[int(time_csv["time"][i]) * 1000: int(time_csv["time"][i + 1]) * 1000]
            result.export(fileName, format='mp3')

        print(">>> >>>", i + 1, "번째 클립 mp3 파일 생성 완료")

    print("\n[lec 생성 종료] mp3 파일 변환 및 mp3 파일 CUT을 종료합니다")


# 메인함수
if __name__ == '__main__':
    time_list = []  # pdf 2 image, 장면 추출 시간, OCR 시간, ORB 유사도 시간
    total_start = time.time()

    # PDF to Image
    tmp_start = time.time()
    pdf2jpg()
    tmp_sec = time.time() - tmp_start
    tmp_times = str(datetime.timedelta(seconds=tmp_sec)).split(".")
    time_list.append(tmp_times[0])

    # 장면전환 추출
    tmp_start = time.time()
    captured_timeline_list = capture_video()
    tmp_sec = time.time() - tmp_start
    tmp_times = str(datetime.timedelta(seconds=tmp_sec)).split(".")
    time_list.append(tmp_times[0])

    # PDF to TXT 저장
    tmp_start = time.time()
    pdf2txt()
    tmp_sec = time.time() - tmp_start
    tmp_times = str(datetime.timedelta(seconds=tmp_sec)).split(".")
    time_list.append(tmp_times[0])

    # 슬라이드와 캡처본 간 이미지 유사도 계산
    tmp_start = time.time()
    tf_timeline_idx = captureFiltering() #캡처 이미지 필터링 # PART 1
    tmp_sec = time.time() - tmp_start
    tmp_times = str(datetime.timedelta(seconds=tmp_sec)).split(".")

    tf_timeline_list = []
    for i, idx_val in enumerate(tf_timeline_idx):
        tf_timeline_list.append(captured_timeline_list[idx_val]) # PART 1
        print("[" + str(i + 1) + "번째 슬라이드 등장시간]", int(captured_timeline_list[idx_val] / 60), "분",
              round(captured_timeline_list[idx_val] % 60), "초")

    df['time'] = tf_timeline_list # PART 1

    selected_slide_list = txtSimCompare() # PART 2
    df['slide'] = selected_slide_list# 해당 전환 시점에 등장한 슬라이드 기입

    df.to_csv(save_path, mode='w')
    time_list.append(tmp_times[0])

    # txt 2 TTS 파일 생성
    tmp_start = time.time()
    txt2TTS()
    tmp_sec = time.time() - tmp_start
    tmp_times = str(datetime.timedelta(seconds=tmp_sec)).split(".")
    time_list.append(tmp_times[0])

    # 원본 강의 영상을 mp3로 변환 및 전환 시점에 맞추어 cut
    tmp_start = time.time()
    cutLectureMp3()
    tmp_sec = time.time() - tmp_start
    tmp_times = str(datetime.timedelta(seconds=tmp_sec)).split(".")
    time_list.append(tmp_times[0])

    total_sec = time.time() - total_start
    total_times = str(datetime.timedelta(seconds=total_sec)).split(".")

    # 장면 추출 시간, PDF to TXT 시간, 이미지 유사도 매칭 시간, CUT 시간, 총 시간
    print("\n■ PDF2JPG 추출 시간:", time_list[0])
    print("■ 장면 추출 시간:", time_list[1])
    print("■ PDF to TXT 시간:", time_list[2])
    print("■ 이미지 유사도 매칭 시간:", time_list[3])
    print("■ TTS 시간:", time_list[4])
    print("■ mp3 변환 및 CUT 시간:", time_list[5])
    print("■□■ 총 소요 시간:", total_times[0])
