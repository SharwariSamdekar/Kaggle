# Image only version

import cv2
# Image to test on
img_file = 'Car_Image.jfif'

# pre-trained car classifier
classifier_file = 'car_detector.xml'



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

# 
print("Code Completed")

