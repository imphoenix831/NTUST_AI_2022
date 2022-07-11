import cv2
import pafy

url = "https://www.youtube.com/watch?v=z_fY1pj1VBw"
#url = "https://luarn.i234.me/background.mp4 "
live = pafy.new(url)
stream = live.getbest(preftype="mp4")

cap = cv2.VideoCapture(stream.url)

#write video format and size
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4', fourcc, 20, (1024, 680))

#while(True):
while (cap.isOpened()):
  success, frame = cap.read()
  frame = cv2.resize(frame, (1024, 680))
  cv2.imshow('AI002 M10609906', frame)

  #read image and write to video file ( out )
  out.write(frame)

  if cv2.waitKey(1) & 0xFF == 27:
    break

#將離開時的影像, 儲存成 output.jpg
cv2.imwrite("output.jpg", frame)

out.release()
cap.release()
cv2.destroyAllWindows()
