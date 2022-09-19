import cv2

vs = cv2.VideoCapture("output.mp4")

while True:
    # `success` is a boolean and `frame` contains the next video frame
    success, frame = vs.read()
    cv2.imshow("frame", frame)
    # wait 20 milliseconds between frames and break the loop if the `q` key is pressed
    if cv2.waitKey(20) == ord('q'):
        break

# we also need to close the video and destroy all Windows
vs.release()
cv2.destroyAllWindows()