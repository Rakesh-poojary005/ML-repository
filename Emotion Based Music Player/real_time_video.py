from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
import pause
import playsound
import random
c=0
count1=0
count2=0
count3=0
count4=0
count5=0
count6=0
count7=0


# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'


face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

cv2.namedWindow('your_face')
camera = cv2.VideoCapture(0)
while True:
    c=c+1
    frame = camera.read()[1]
    # reading the frame
    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                            flags=cv2.CASCADE_SCALE_IMAGE)

    canvas = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = frame.copy()
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
                       key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
        # the ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        print("emotion_probability",emotion_probability)
        label = EMOTIONS[preds.argmax()]
        print("label",label)
        if label=="happy":
            print("Happy")
            L = ['love.mp3', 'Happy1.mp3', 'Happy2.mp3', 'Happy3.mp3']
            S = random.randint(0, 3)
            print("S", S)
            playsound.playsound(L[S], True)

            count1=count1+1
        elif label=="angry":
            #playsound.playsound('love.mp3', True)
            print("angry")
            L = ['angry.mp3', 'angry1.mp3', 'angry2.mp3', 'angry3.mp3']
            S = random.randint(0, 3)
            print("S", S)
            playsound.playsound(L[S], True)
            count2 = count2 + 1
        elif label == "surprised":
            print("surprised")
            L = ['Surprise1.mp3', 'Surprise2.mp3', 'angry2.mp3', 'angry3.mp3']
            S = random.randint(0, 3)
            print("S", S)
            playsound.playsound(L[S], True)
            count3 = count3 + 1
        elif label=="disgust":
            print("disgust")
            L = ['Disgust1.mp3', 'Disgust2.mp3', 'Disgust3.mp3', 'Disgust4.mp3']
            S = random.randint(0, 3)
            print("S", S)
            playsound.playsound(L[S], True)
            count4 = count4 + 1
        elif label=="sad":
            print("sad")
            L = ['sad.mp3', 'sad1.mp3', 'sad2.mp3', 'sad3.mp3']
            S = random.randint(0, 3)
            print("S", S)
            playsound.playsound(L[S], True)
            count5 = count5 + 1
        elif label=="scared":
            print("scared")
            L = ['Scare1.mp3', 'Scare2.mp3', 'Scare3.mp3', 'Scare4.mp3']
            S = random.randint(0, 3)
            playsound.playsound(L[S], True)
            count6 = count6 + 1
        elif label=="neutral":
            print("neutral")
            L = ['m11.mp3', 'Neutral.mp3', 'Neutral1.mp3', 'Neutral2.mp3']
            S = random.randint(0, 3)
            print("S",S)
            playsound.playsound(L[S], True)

            count7 = count7 + 1

    else:
        continue

    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
        # construct the label text
        text = "{}: {:.2f}%".format(emotion, prob * 100)

        # draw the label + probability bar on the canvas
        # emoji_face = feelings_faces[np.argmax(preds)]


        w = int(prob * 300)
        cv2.rectangle(canvas, (7, (i * 35) + 5),
                      (w, (i * 35) + 35), (0, 0, 255), -1)
        cv2.putText(canvas, text, (10, (i * 35) + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 2)
        cv2.putText(frameClone, label, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                      (0, 0, 255), 2)
    # for c in range(0, 3):
    #        frame[200:320, 10:130, c] = emoji_face[:, :, c] * \
    #        (emoji_face[:, :, 3] / 255.0) + frame[200:320,
    #        10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)



        if(emotion=="happy"):
           # playsound.playsound('m11.mp3', True)
            pause.seconds(0.5)


        elif (emotion == "disgust"):
           # playsound.playsound('m11.mp3', True)
            pause.seconds(0.5)

            # print("DIGUSTING")
        elif (emotion == "scared"):
            #playsound.playsound('m11.mp3', True)
            pause.seconds(0.5)

            # print("SCARRY")
        elif (emotion == "angry"):
           # playsound.playsound('m11.mp3', True)
            pause.seconds(0.5)

            # print("ANGRYYY")
        elif (emotion == "sad"):
            pause.seconds(0.5)


            # print("SAD")
        elif (emotion == "surprised"):
           # playsound.playsound('m11.mp3', True)
            pause.seconds(0.5)

        elif (emotion == "neutral"):
            #playsound.playsound('m11.mp3', True)
            pause.seconds(0.5)



    cv2.imshow('your_face', frameClone)
    cv2.imshow("Probabilities", canvas)
    cv2.waitKey(1)
    print("c",c)
    if c > 20:
        c=0
    if c > 4:
        break
    #if cv2.waitKey(1) & 0xFF == ord('q'):
     #   break
print("Happy",count1)
print("Angry",count2)
print("Surprised",count3)
print("Disgust",count4)
print("Sad",count5)
print("Scared",count6)
print("Neutral",count7)
"""
if count1 > 3 and count7 > 2:
    print("Candidate is well Prepared")
elif count1 < 2 and count5 > 2:
    print("Candidate is not Prepared and Sad")
"""


camera.release()
cv2.destroyAllWindows()














