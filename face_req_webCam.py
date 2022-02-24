import pyttsx3 as tts
import threading
import cv2
from fr_utils import *


speaker = tts.init()

def swift_speak(name):
    voices = speaker.getProperty('voices')
    rate = speaker.getProperty('rate')
    speaker.setProperty('rate', 150)
    speaker.setProperty('voice', voices[1].id)
    speaker.say(name)
    speaker.runAndWait()



speak = True
currentName = ''
prevName = ''

def speakName():
    global speak
    global currentName

    while True:
        if speak:
            swift_speak(currentName)
            speak = False
    

x = threading.Thread(target=speakName, daemon=True)
x.start()



def main():
    
    global currentName, prevName, speak
    cam = cv2.VideoCapture(0)
    cTime = 0
    pTime = 0
        
    while True:
        _, frame = cam.read()
        
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        small_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        landmarks = face_recognition.face_landmarks(small_rgb)
        
        
        cTime = time.time()
        fps = 1 / (cTime-pTime)
        pTime = cTime
        
        for landmark in landmarks:
            leftEye = landmark['left_eye']
            rightEye = landmark['right_eye']
            
            # Get aspect ratios for left and right eyes
            leftEar = get_ear(leftEye)
            rightEar = get_ear(rightEye)
            
            ear = (leftEar + rightEar) / 2.0

            if speak == False:
                currentName = recognize(small_rgb, ear,frame)
                if currentName != prevName:
                    speak = True
                prevName = currentName
                
        cv2.putText(frame, f'{int(fps)} fps', (500, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255, 255), 1)   
        cv2.imshow('video', frame)
        if cv2.waitKey(1) == ord('q'):
            break
        

    cam.release()
    cv2.destroyAllWindows()   


if __name__ == '__main__':
    main()

