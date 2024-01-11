import cv2

# Initialize the video capture using the camera index (0 for the default camera)
cap = cv2.VideoCapture(0)

# Load the car cascade classifier
car_cascade = cv2.CascadeClassifier('car.xml')

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cars in the frame using the cascade classifier
    cars = car_cascade.detectMultiScale(gray, 1.1, 9)

    for (x, y, w, h) in cars:
        # Draw a rectangle around the detected car
        cv2.rectangle(frame, (x, y), (x + w, y + h), (51, 51, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (51, 51, 255), -2)
        cv2.putText(frame, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('car', frame[y:y + h, x:x + w])

    # Display the frame with car detections
    cv2.imshow('Car Detection System', frame)

    # Check for the 'Esc' key to exit the loop
    if cv2.waitKey(1) == 27:  # 1 millisecond delay, Esc key
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
