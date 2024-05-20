import cv2
import numpy as np
import imutils
from os import listdir
from os.path import isfile, join

person_name ="Godwin"
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
eye_casade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')

def face_extractor(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is():
        return None

    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h+50, x:x+w+10]

    return cropped_face

cap = cv2.VideoCapture(0)
count = 0
name = person_name

while True:

    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        cv2.imwrite('D:/trails/'+ str(count) + '.jpg', face)

        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper', face)

    else:
        print("Face not found")
        pass

    if cv2.waitKey(1) == "q" or count == 50: 
        break

cap.release()
cv2.destroyAllWindows()
print("Collecting Samples Complete")

data_path = "D:/trails/"
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

Training_Data, Labels = [],[]

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)
model = cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model trained sucessefully")


face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
eye_casade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')

def face_detector(img, size=0.5):

    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    frame=imutils.resize(frame,width=1350)

    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        results = model.predict(face)

        if results[1] < 500:
            confidence = int( 100 * (1 - (results[1])/400) )
            display_string = str(confidence) + '% Confident it is '+ name

        cv2.putText(image, display_string, (200, 120), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 2)

        if confidence > 80:
            cv2.putText(image, "Unlocked", (900, 450), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 2)
            cv2.imshow('Face Recognition', image )
        else:
            cv2.putText(image, "Locked", (900, 450), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 2)
            cv2.imshow('Face Recognition', image )

    except:
        cv2.putText(image, "No Face Found", (200, 120) , cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 2)
        cv2.putText(image, "Locked", (900, 450), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 2)
        cv2.imshow('Face Recognition', image )
        pass

    if cv2.waitKey(1) == "q": 
        break

cap.release()
cv2.destroyAllWindows()
