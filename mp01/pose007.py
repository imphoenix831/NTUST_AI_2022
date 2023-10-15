import cv2
import mediapipe as mp
import numpy as np
import math
import pafy
import os
import shutil
from time import strftime

from PIL import ImageFont, ImageDraw, Image
from moviepy.editor import *

from ultralytics import YOLO


# 定義加入文字函式
def putText(x,y,text,size=20,color=(0,0,0)):
    global frame
    fontpath = 'NotoSansTC-Regular.otf'            # 字型
    font = ImageFont.truetype(fontpath, size)      # 定義字型與文字大小
    imgPil = Image.fromarray(frame)                  # 轉換成 PIL 影像物件
    draw = ImageDraw.Draw(imgPil)                  # 定義繪圖物件
    draw.text((x, y), text, fill=color, font=font) # 加入文字
    print("x:{},y:{},text:{}".format(x,y,text))
    frame = np.array(imgPil)                         # 轉換成 np.array



def pose_detect(xx1, x1,x2,x3,y1,y2,y3,switch,count):
    color = (0, 0, 255)

    #cv2.line(img, pt1=起始點座標, pt2:結束點座標, color=顏色, thickness: 線條粗細
    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
    cv2.line(frame, (x3, y3), (x2, y2), (0, 255, 255), 3)

    #cv2.circle(img, center=中心點座標, radius: 半徑, color , thickness )
    cv2.circle(frame, (x1, y1), 10, (0, 255, 255), cv2.FILLED)
    cv2.circle(frame, (x1, y1), 15, (0, 255, 255), 2)
    cv2.circle(frame, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
    cv2.circle(frame, (x2, y2), 15, (0, 0, 255), 2)
    cv2.circle(frame, (x3, y3), 10, (0, 255, 255), cv2.FILLED)
    cv2.circle(frame, (x3, y3), 15, (0, 255, 255), 2)

    pose_angle = abs(int(math.degrees(math.atan2(y1 - y2, x1 - x2) - math.atan2(y3 - y2, x3 - x2))))
    # 以10到170度 來計算右手彎曲的程度，最高%=100，最低%=0
    pose_per = np.interp(pose_angle, (10, 170), (100, 0))

    # 根據右手彎曲程度計算bar的高度 Y軸座標，最高y=200，最低y=400
    pose_bar = int(np.interp(pose_angle, (10, 170), (200, 400)))

    # 畫矩形來代表bar的高度， 同時印出數字
    #cv2.rectangle(img, pt1:左上座標 , pt2:右下座標, color, thickness )
    cv2.rectangle(frame, (xx1, int(pose_bar)), (xx1 + 30, 400), color, cv2.FILLED)

    #cv2.putText(img, text:文字內容, org 文字座標,  fontFace:文字字型, fontscale: 文字尺寸, color, thickness, linetype: 外框線條樣式
    cv2.putText(frame, str(int(pose_per)) + '%', (xx1 - 10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # 手起到95%或5%算半個
    color = (0, 0, 255)
    if pose_per >= 95:
        color = (0, 255, 0)
        if switch == 0:
            count += 0.5
            switch = 1
    if pose_per <= 5:
        color = (0, 255, 0)
        if switch == 1:
            count += 0.5
            switch = 0

    cv2.putText(frame, str(count), (xx1 - 40, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)
    cv2.imshow('MediaPipe Pose Workout', frame)

    return switch, count

def covert_mov_to_mp4(source_type,url_filename, video_type, output_filename):

    video = VideoFileClip(url_filename)   #讀取影片
    format_list =['mov']   # 要轉換的格式清單

    for i in format_list:
        output = video.copy()
        output.write_videofile(f"output.{i}", temp_audiofile=output_filename, remove_temp=True, codec="libx264",
                           audio_codec="aac")

    print('covert_mov_to_mp4 ok')


def pose_detect_write_to_file(source_type,url_filename, video_type, output_filename):
    # pose = mp.solutions.pose.Pose()
    # ENABLE_SEGMENTATION: 去背 ;
    pose = mp.solutions.pose.Pose(model_complexity=1, smooth_landmarks=True, smooth_segmentation=True,
                                  enable_segmentation=False, min_detection_confidence=0.8, min_tracking_confidence=0.5)
    # 連結 Pose 間的點, 若要畫半身, 可以自己建 半身的 list
    conn = mp.solutions.pose.POSE_CONNECTIONS
    # print("Conn:{}".format(conn))

    # 把 pose 中的點和線畫出來
    mp_drawing = mp.solutions.drawing_utils

    # pose 中的點和線 的顏色和大小
    # 使用官方的點.線 style
    # spec = mp.solutions.drawing_styles.get_default_pose_landmarks_style()

    # 設定個人化的線
    spec = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=3, circle_radius=1)

    switch, count = 0, 0
    switch1, count1 = 0, 0
    play = 1

    # 盜墓筆記 The Lost Tomb Season1 第10集
    # url = 'https://www.youtube.com/watch?v=CfJYS7-QfYc'
    # 跑步姿勢
    # url = 'https://www.youtube.com/watch?v=Myekr_6F2aw'

    # 陳式太極拳

    # url = 'https://youtu.be/7ra_tCmSq9g'
    # url = 'https://youtube.com/shorts/0bjq6SVm2Fk?feature=shared'

    if source_type =='url':
        # read video from URL
        live = pafy.new(url_filename)
        stream = live.getbest(preftype="mp4")
        cap = cv2.VideoCapture(stream.url)

    elif source_type =='camera':
        cap = cv2.VideoCapture(0)

    elif source_type =='file':
        cap = cv2.VideoCapture(url_filename)

    # write video format and size
    if video_type =='mp4':
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_filename = output_filename

    elif video_type =='avi':
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_filename = output_filename


    frame_width = int (cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_hieght = int (cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate  = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(out_filename, fourcc, frame_rate, (frame_width,frame_hieght ))

    while cap.isOpened():
        if play:
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)
            h, w, c = frame.shape

            ## xx1 計算 bar 起始位置 : 0.1
            xx1 = int(w * 0.1)
            xx2 = int(w * 0.8)
            poslist = []

            # 如果有偵測到 pose 時 , results.pose_landmarks 共 0-33 , 共 34 個點
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, conn, spec)
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    poslist.append([id, cx, cy])

            try:
                # 右手肘的角度 ; 12: 右肩  14:右手臂; 16:右上臂
                x1, y1 = poslist[12][1], poslist[12][2]
                x2, y2 = poslist[14][1], poslist[14][2]
                x3, y3 = poslist[16][1], poslist[16][2]

                switch, count = pose_detect(xx1, x1, x2, x3, y1, y2, y3, switch, count)

                # 左手肘的角度 ; 11: 左肩  13:左手臂; 15:左上臂
                lx1, ly1 = poslist[11][1], poslist[11][2]
                lx2, ly2 = poslist[13][1], poslist[13][2]
                lx3, ly3 = poslist[15][1], poslist[15][2]
                switch1, count1 = pose_detect(xx2, lx1, lx2, lx3, ly1, ly2, ly3, switch1, count1)


            except:
                pass

            out.write(frame)
            cv2.imshow(output_filename, frame)

            key = cv2.waitKey(10)
            # print('key:'.format(key))

            if key == ord('q') or key == 27:
                print('break')
                # 將偵測後手的影片儲存起來
                break

            elif key == 13:   # play is not working
                print("play:".format(play))
                if play != 1:
                    putText(100, 100, '暫停')  # 放入文字

                print('play / pause')
                play = play ^ 1

                pass
            elif key == 32:
                systime = strftime("%Y%m%d%H%M%S")
                imgname = os.path.join('images/photo-' + systime + '.jpg')
                putText(100, 10, imgname)  # 放入文字
                cv2.imwrite(imgname, frame)

    out.release()
    cap.release()
    cv2.destroyAllWindows()
    print('pose_detect_write_to_file to {}'.format(output_filename))



def merge_two_video_side_by_side(Video1_filename, Video2_filename , video_type, output_filename):
    #open the first video file
    video1 = cv2.VideoCapture(Video1_filename)

    #open the second video file
    video2 = cv2.VideoCapture(Video2_filename)

    #Get the frame size and frame rate of the first video
    frame_width = int (video1.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_hieght = int (video1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate  = int(video1.get(cv2.CAP_PROP_FPS))

    #Create a VideoWriter object to write the output video

    if video_type =='mp4':
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out_filename = output_filename
    elif video_type =='avi':
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_filename = output_filename

    out = cv2.VideoWriter(out_filename, fourcc, frame_rate, (frame_width *2,frame_hieght ))

    #Loop through the frame of both videos and write them to the output video
    while True:
        # Read a frame from the first video
        ret1, frame1 = video1.read()

        # Read a frame from the seconed video
        ret2, frame2 = video2.read()

        #stop the loop if either video has reach its end
        if not ret1 or not ret2:
            break

        # Resize the Second video to match the size of the first video
        frame2 = cv2.resize(frame2, ( frame_width,frame_hieght ))

        # Create a black canvas to place the merge frame on
        canvas = np.zeros((frame_hieght,frame_width * 2, 3 ) , dtype=np.uint8)

        # place the first frame on the left side of the canvas
        canvas[:, :frame_width] = frame1

        # place the second frame on the right of the canvas
        canvas[:,frame_width: ] = frame2

        # write the merge frame to the output video
        out.write ( canvas)

    video1.release()
    video2.release()
    out.release()
    print('merge_two_video_side_by_side to {}'.format(output_filename))


def video_get_audio(source_type,url_filename, video_type, output_filename):
    video = VideoFileClip(url_filename)    # 讀取影片
    audio = video.audio                    # 讀取影片的聲音
    audio.write_audiofile(output_filename)
    print('video get audio and write to {}'.format(output_filename))

def video_with_audio(video_filename, audio_filename, output_filename):
    video = VideoFileClip(video_filename)    # 讀取影片
    audio = AudioFileClip(audio_filename)    # 讀取音檔
    output = video.set_audio(audio)         # 合併影片與聲音
    output.write_videofile(output_filename,temp_audiofile="temp-audio.m4a",remove_temp=True,codec="libx264", audio_codec="aac")
    print('video_with_audio and write to {}'.format(output_filename))

def remove_unused_file(Folder_Path,source_file):
    if os.path.exists(Folder_Path):
        merge_file_name   = Folder_Path + '/' + source_file + '_Merge.mp4'
        detect_file_name  =  Folder_Path + '/' + source_file+'_AI.mp4'
        Audio_file_name   = Folder_Path + '/' + source_file + '_Audio.mp3'

        if os.path.isfile(merge_file_name):
            os.remove(merge_file_name)
            print("Remove Merge File:{}".format(merge_file_name))

        if os.path.isfile(Audio_file_name):
            os.remove(Audio_file_name)
            print("Remove Audio File:{}".format(Audio_file_name))

        if os.path.isfile(detect_file_name):
            os.remove(detect_file_name)
            print("Remove Audio File:{}".format(detect_file_name))







if __name__ == '__main__':
    Func = 4

    Folder_name       =  '20230926_金剛搗捶_抱虎歸山'
    Folder_Path       =  'D:/4-KM_Map/台科大EMBA/1-課程/11202/陳式太極拳/' + Folder_name
    predict_folder    =  'output/predict/'

    source_file       =  '金剛搗捶_抱虎歸山_20230926-2'   # source with audio
    source2_file      =  '陳式太極拳_金剛搗捶_抱虎歸山_20230926.mp4'

    predict_file_name =  predict_folder +source_file+'.avi'
    source_file_name  =  Folder_Path + '/' + source_file +'.mov'
    source2_file_name =  Folder_Path + '/' + source2_file

    detect_file_name  =  Folder_Path + '/' + source_file+'_AI.mp4'
    merge_file_name   =  Folder_Path + '/' + source_file+'_Merge.mp4'
    Audio_file_name   =  Folder_Path + '/' + source_file+'_Audio.mp3'

    AI_file_name      =  Folder_Path + '/' + '陳式太極拳_'+ source_file+'_AI_Audio.mp4'
    output_file_name  =  Folder_Path + '/' + '陳式太極拳_'+ source_file +'_All.mp4'

    ########################################################################

    if not os.path.exists(Folder_Path):
        os.makedirs(Folder_Path)

    #########################################################################
    #covert_mov_to_mp4('mov', '20231004.mov', 'mp4', '20231004.mp4'):

    if Func == 1 :
        # Source with AI and Audio and  merge source
        pose_detect_write_to_file('file', source_file_name, 'mp4', detect_file_name )

        merge_two_video_side_by_side(source_file_name, detect_file_name, 'mp4', merge_file_name)
        video_get_audio('mp4',source_file_name, 'mp3', Audio_file_name)
        video_with_audio(merge_file_name,  Audio_file_name, output_file_name)
        video_with_audio(detect_file_name, Audio_file_name, AI_file_name)
        remove_unused_file(Folder_Path,source_file)



    elif Func == 2 :
        # Source with AI and Audio without merge source

        pose_detect_write_to_file('file', source_file_name, 'mp4', detect_file_name)
        video_get_audio('mp4',source_file_name, 'mp3', Audio_file_name)
        video_with_audio(detect_file_name, Audio_file_name, AI_file_name)
        remove_unused_file(Folder_Path,source_file)


        #shutil.move(output_file_name, output_file_name + '/' + output_file_name)

        ##########################################################################################
    elif Func == 3:
        # two Source merge with source Audio
        print('Source: {} , Source2:{}'.format(source_file_name,source2_file_name))

        merge_two_video_side_by_side(source_file_name, source2_file_name, 'mp4', merge_file_name)
        video_get_audio('mp4',source_file_name, 'mp3', Audio_file_name)
        video_with_audio(merge_file_name, Audio_file_name, output_file_name)
        remove_unused_file(Folder_Path,source_file)

    elif Func == 4:
        model = YOLO('yolov8n-pose.pt')  # load a pretrained model (recommended for training)
        # Run inference on the source
        results = model(source=source_file_name, show=True, conf=0.8, save=True, show_conf=False, boxes=False,line_width=1, imgsz=1024, project='output')

        merge_two_video_side_by_side(source_file_name, predict_file_name, 'mp4', merge_file_name)
        video_get_audio('mp4', source_file_name, 'mp3', Audio_file_name)
        video_with_audio(predict_file_name, Audio_file_name, AI_file_name)
        video_with_audio(merge_file_name, Audio_file_name, output_file_name)
        remove_unused_file(Folder_Path, source_file)


    else:
        print ('請指定功能')


    print('Finshed....YA!!!  Write to  {}'.format(output_file_name))


