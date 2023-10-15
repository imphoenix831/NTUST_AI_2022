import cv2
import mediapipe as mp
import numpy as np
import math
import pafy
import os
from time import strftime

from PIL import ImageFont, ImageDraw, Image

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

    if video_type ='mp4'
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out_filename = output_filename+'.mp4''
    elif video_type ='avi'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_filename = output_filename+'.avi''

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












if __name__ == '__main__':
    # pose = mp.solutions.pose.Pose()
    # ENABLE_SEGMENTATION: 去背 ;
    pose = mp.solutions.pose.Pose(model_complexity=1, smooth_landmarks=True, smooth_segmentation=True,
                                  enable_segmentation=False, min_detection_confidence=0.8, min_tracking_confidence=0.5)
    # 連結 Pose 間的點, 若要畫半身, 可以自己建 半身的 list

    conn = mp.solutions.pose.POSE_CONNECTIONS
    #print("Conn:{}".format(conn))

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
    #url = 'https://www.youtube.com/watch?v=CfJYS7-QfYc'
    #跑步姿勢
    #url = 'https://www.youtube.com/watch?v=Myekr_6F2aw'

    #陳式太極拳

    #url = 'https://youtu.be/7ra_tCmSq9g'
    #url = 'https://youtube.com/shorts/0bjq6SVm2Fk?feature=shared'

    #live = pafy.new(url)
    #stream = live.getbest(preftype="mp4")

    #cap = cv2.VideoCapture(0)

    cap = cv2.VideoCapture('sister.mp4')

    # read video from URL
    #cap = cv2.VideoCapture(stream.url)
    # write video format and size

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    fps = cap.get(cv2.CAP_PROP_FPS)

    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    #out = cv2.VideoWriter('Tues01.mp4', fourcc, 15, (1024, 680))
    out = cv2.VideoWriter('Tues02.mp4', fourcc, 15, (int(w), int(h)))

    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter('output2.avi', fourcc, 20.0, (640, 480))


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

            #如果有偵測到 pose 時 , results.pose_landmarks 共 0-33 , 共 34 個點
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
            cv2.imshow('永和社區大學 陳式太極拳 陳駿巃師父 ', frame)

            key = cv2.waitKey(10)
            #print('key:'.format(key))

            if key == ord('q') or key == 27:
                print('break')
                # 將偵測後手的影片儲存起來
                break

            elif key == 13:
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
