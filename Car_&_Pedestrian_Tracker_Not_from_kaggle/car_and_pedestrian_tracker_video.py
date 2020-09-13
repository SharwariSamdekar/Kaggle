# Image only version

import cv2
import pafy 
import youtube_dl
# Code to capture the video from youtube
# pafy and youtube_dl modules are required

# url = 'https://www.youtube.com/watch?v=d4L1Pte7zVc'
url = 'https://www.youtube.com/watch?v=WriuvU1rXkc'
vPafy = pafy.new(url)
play = vPafy.getbest()

print(play.url)
video = cv2.VideoCapture(play.url)


# pre-trained car classifier
car_tracker_file = 'car_detector.xml'
pedestrian_tracker_file = 'haarcascade_fullbody.xml'

# Create car classifier
# cascade cause we are using harr cascade algorithm
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)


while True :

    # Read the current frame
    (read_successful, frame) = video.read()

    # Safe Coding
    if read_successful :
        # Must convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else :
        break

    # detect cars and pedestrians
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    # Draw rectangles around the cars
    # car1 = cars[0]
    for (x, y, w, h) in cars :
        cv2.rectangle(frame,(x,y),(x+1, y+1), (255, 0, 0), 2)
        cv2.rectangle(frame,(x,y),(x+w, y+h), (0, 0, 255), 2)

    # Draw rectangles around the cars

    for (x, y, w, h) in pedestrians :
        cv2.rectangle(frame,(x,y),(x+w, y+h), (0, 255, 255), 2)

    # Disply the frame
    cv2.imshow('Car Detector', frame)

    # Don't Autoclose
    key = cv2.waitKey(1)


    # Stop if Q key is pressed 
    if key == 81 or key == 113 :
        break

# Release the VideoCapture object 

video.release()


"""



# Create opencv image
img = cv2.imread(img_file)

# Create car classifier
# cascade cause we are using harr cascade algorithm
car_tracker = cv2.CascadeClassifier(classifier_file)

# convert to grayscale (needed for haar cascade)
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

 
# detect cars
cars = car_tracker.detectMultiScale(black_n_white)

# Draw rectangles around the cars
# car1 = cars[0]
for (x, y, w, h) in cars :
    cv2.rectangle(img,(x,y),(x+w, y+h), (0, 0, 255), 2)

# Display the image with car spotted
cv2.imshow("Car Detector", img)

# Don't autoclose
cv2.waitKey()

# """
print("Code Completed")

