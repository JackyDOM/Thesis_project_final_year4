import cv2
import mediapipe as mp
import pyautogui
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import time

# Initialize MediaPipe Hand
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to count raised fingers
def count_fingers(hand_landmarks):
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
    thumb_tip = 4
    fingers_up = 0

    # Check fingers
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers_up += 1

    # Check thumb
    if hand_landmarks.landmark[thumb_tip].x > hand_landmarks.landmark[thumb_tip - 2].x:
        fingers_up += 1  # Thumb considered "up"

    return fingers_up

# Start Selenium WebDriver to open the game
service = Service(ChromeDriverManager().install())  # Automatically manage the ChromeDriver
driver = webdriver.Chrome(service=service)

driver.get('https://poki.com/en/g/subway-surfers')
time.sleep(3)  # Wait for the game to load

# Optionally, click "Play" or any start button (ensure you update XPath if needed)
try:
    play_button = driver.find_element("xpath", '//button[contains(text(), "Play")]')  # Adjust the XPath
    play_button.click()
except Exception as e:
    print(f"Error finding Play button: {e}")

# Open Camera for Hand Gesture Control
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingers = count_fingers(hand_landmarks)

            # Gesture-based Controls for Subway Surfers
            if fingers == 1:
                # Right gesture
                pyautogui.press("right")  # Move Right
                driver.find_element("body").send_keys(Keys.RIGHT)  # Simulate Right Arrow key press
            elif fingers == 2:
                # Left gesture
                pyautogui.press("left")   # Move Left
                driver.find_element("body").send_keys(Keys.LEFT)  # Simulate Left Arrow key press
            elif fingers == 3:
                # Up gesture (Jump)
                pyautogui.press("up")     # Jump (Up)
                driver.find_element("body").send_keys(Keys.UP)    # Simulate Up Arrow key press
            elif fingers == 4:
                # Down gesture (Duck)
                pyautogui.press("down")   # Duck (Down)
                driver.find_element("body").send_keys(Keys.DOWN)  # Simulate Down Arrow key press
            elif fingers == 5:
                # Game start gesture
                print("Game Starting!")  # This can trigger game start logic if needed

    cv2.imshow("Hand Gesture Controller", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
driver.quit()
