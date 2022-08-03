import cv2

def nothing(x):
    pass

img1 = cv2.imread('data/1.png')
img2 = cv2.imread('data/2.png')
img1 = cv2.resize(img1, (640, 480))
img2 = cv2.resize(img2, (640, 480))

#產生 new windows
cv2.namedWindow('cv1')
cv2.createTrackbar('%', 'cv1', 0, 100, nothing)

while (True):
    r = cv2.getTrackbarPos('%', 'cv1')
    r = float(r) / 100.0
    img = cv2.addWeighted(img1, r, img2, 1.0 - r, 0)
    cv2.putText(img, str(r), (3, 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    #cv2.imshow('merged image', img)

    cv2.imshow('cv1', img) #若是 cv1 , img 放在跟  trackbar cv1 視窗在一起
    if cv2.waitKey(5) & 0xFF == 27:
        break
cv2.destroyAllWindows()
