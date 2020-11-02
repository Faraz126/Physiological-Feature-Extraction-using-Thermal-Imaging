import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
video_capture_2 = cv2.VideoCapture('video.avi')
anterior = 0


model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
model.load_weights('model.h5')
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

SIZE_X = int(1024 / 8)
SIZE_Y = int(768 / 8)

model3 = Sequential()
model3.add(Conv2D(32, (2, 2), padding = 'same', activation='tanh', input_shape=(SIZE_Y, SIZE_X, 1)))

model3.add(Conv2D(32, (2, 2), activation='relu'))
model3.add(MaxPooling2D(pool_size=(2, 2)))
#model3.add(Dropout(0.5))
model3.add(Conv2D(64, (3, 3), activation='tanh'))
#model3.add(Dropout(0.5))
model3.add(Conv2D(64, (3, 3), activation='tanh'))
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(Conv2D(128, (3, 3), activation='relu'))
model3.add(Conv2D(128, (3, 3), activation='tanh'))
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(Conv2D(256, (3, 3), activation='tanh'))
model3.add(Conv2D(256, (5, 5), activation='relu'))
model3.add(Dropout(0.5))

model3.add(Flatten())
model3.add(Dense(512, activation='relu'))
#model3.add(Dropout(0.5))
model3.add(Dense(4, activation='sigmoid'))


model3.load_weights('./modelnew_augmentedData_best.h5')

model5 = Sequential()
model5.add(Conv2D(32, (2, 2), padding = 'same', activation='tanh', input_shape=(SIZE_Y, SIZE_X, 1)))

model5.add(Conv2D(32, (2, 2), activation='relu'))
model5.add(MaxPooling2D(pool_size=(2, 2)))
model5.add(Dropout(0.5))
model5.add(Conv2D(64, (3, 3), activation='tanh'))
model5.add(Dropout(0.5))
model5.add(Conv2D(64, (3, 3), activation='tanh'))
model5.add(MaxPooling2D(pool_size=(2, 2)))
model5.add(Conv2D(128, (3, 3), activation='relu'))
model5.add(Conv2D(128, (3, 3), activation='tanh'))
model5.add(MaxPooling2D(pool_size=(2, 2)))

model5.add(Flatten())
model5.add(Dense(512, activation='relu'))
#model3.add(Dropout(0.5))
model5.add(Dense(68 * 2, activation='sigmoid'))

model5.load_weights('./model3_keypoints_augmentedData_best.h5')




while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    ret, thermal_frame = video_capture_2.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


    # Display the resulting frame

    thermal_frame_resized = cv2.resize(thermal_frame, (frame.shape[1], frame.shape[0]))

    gray = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)
    thermal_frame_small = cv2.resize(gray / 255, (128, 96))



    prediction = model3.predict(np.reshape(thermal_frame_small, (1, 96, 128, 1)))[0]
    x = int(prediction[0] * frame.shape[1])
    w = int((prediction[1] * frame.shape[1]))
    y = int(prediction[2] * frame.shape[0])
    h = int((prediction[3] * frame.shape[0]))

    cv2.rectangle(thermal_frame_resized, (x, y), (w, h), (0, 255, 0), 2)

    prediction = model5.predict(np.reshape(thermal_frame_small, (1, 96, 128, 1)))[0]

    x = prediction[:68]
    y = prediction[68:]
    for i in range(68):
        cv2.rectangle(thermal_frame_resized, (int(x[i] * frame.shape[1]), int(y[i] * frame.shape[0])), (int(x[i] * frame.shape[1]) + 2, int(y[i] * frame.shape[0]) + 2), (255, 255, 0), 2)



    new_frame = np.concatenate((frame, thermal_frame_resized), axis = 1)

    cv2.imshow('Video', new_frame)
    #print(new_frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    #cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()