import cv2

cascade_classifier = cv2.CascadeClassifier(r'C:\Users\meena\OneDrive\Desktop\Hackathon\haarcascade_frontalface_default.xml')
#cascade_classifier = cv2.CascadeClassifier(r'C:\Users\meena\OneDrive\Desktop\Hackathon\haarcascade_eye.xml')

cap = cv2.VideoCapture(r'C:\Users\meena\OneDrive\Desktop\Hackathon\WhatsApp Video 2024-03-10 at 11.52.00_ce169a77.mp4')

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detections = cascade_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in detections:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()