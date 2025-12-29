import cvzone as cvz
import cv2 
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from google import genai
import PIL.Image as Image
#not the best way to store api key but for demo purpose it's fine
client = genai.Client(api_key="AIzaSyAiuO9BNelRZ854jodGLH_DVlnsig7GQO8")
# model=genai.GenerativeModel("gemini-2.5-flash")


# Initialize the webcam to capture video
# The '2' indicates the third camera connected to your computer; '0' would usually refer to the built-in camera
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
prev_position = None
canvas = None
image_combo = None
# Initialize the HandDetector class with the given parameters
detector = HandDetector(mode=False, maxHands=1, detectionCon=0.7, minTrackCon=0.5)



def getHandInfor(img):
    hand, img = detector.findHands(img, draw=True, flipType=True)
    # Check if any hands are detected
    if hand:
        # Information for the first hand detected
        hand = hand[0]  # Get the first hand detected
        lmList = hand["lmList"]  # List of 21 landmarks for the first hand
        # bbox1 = hand["bbox"]  # Bounding box around the first hand (x,y,w,h coordinates) __ not needed 
        # center1 = hand['center']  # Center coordinates of the first hand
        # handType1 = hand["type"]  # Type of the first hand ("Left" or "Right")
        # Count the number of fingers up for the first hand
        fingers1 = detector.fingersUp(hand)
        print(fingers1)
        return fingers1, lmList
    else:
        return None

def draw(info, prev_position,canvas):
    fingers, lmList = info
    # Initialize currentPoint to None if it didn't detect anything then the previous point will become none 
    currentPoint = None
    if fingers == [0,1,0,0,0]:
        #draw line between the previous and the new point over the image
        #we need x and y coordinates so we get x and y of point 8 
        currentPoint = lmList[8][0:2]
        #if previous position is none then draw a point 
        if prev_position is None:
            prev_position = currentPoint
        #draw on the canvas 
        cv2.line(canvas, currentPoint, prev_position, (255,0,255), 10)

    elif fingers == [1,1,1,1,1]:
        canvas = np.zeros_like(img)
    return currentPoint, canvas

def sendToAi(client, fingers):
    if fingers == [1,1,1,1,0]:
        pil_image = Image.fromarray(canvas)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=["Solve the math problem shown in the image.",pil_image
            ]
        )
        print(response.text)

# Continuously get frames from the webcam
while True:
    # Capture each frame from the webcam
    # 'success' will be True if the frame is successfully captured, 'img' will contain the frame
    success, img = cap.read()
    #flipping it horizontally for a mirror effect
    img = cv2.flip(img, 1)
    #creating a new canvas to draw on it
    
    if canvas is None or canvas.shape != img.shape:
        canvas = np.zeros_like(img)
        prev_position = None

    info = getHandInfor(img)
    if info:
        fingers, lmList = info
        print(fingers)
        prev_position, canvas = draw(info, prev_position, canvas)
        sendToAi(client, fingers)

    # print("img shape:", img.shape)
    # print("canvas shape:", canvas.shape)    
    image_combo = cv2.addWeighted(img,0.7,canvas,0.3,0)

    # Find hands in the current frame
    # The 'draw' parameter draws landmarks and hand outlines on the image if set to True
    # The 'flipType' parameter flips the image, making it easier for some detections
    

    # Display the image in a window 
    cv2.imshow("Image", img)
    # cv2.imshow("Canvas", canvas)
    cv2.imshow("Image", image_combo)

    
    # Keep the window open and update it for each frame; wait for 1 millisecond between frames
    cv2.waitKey(1)


