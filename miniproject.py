import cv2
import mediapipe as mp
# Initialize Mediapipe hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)  # Detect one hand at a time
mp_drawing = mp.solutions.drawing_utils
# Function to detect specific gestures
def detect_gesture(hand_landmarks):
    # Relevant landmark points
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_dip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_dip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_dip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]

    # Gesture 1: Shambhala Mudra (Thumb touching Ring Finger)
    thumb_ring_touch = abs(thumb_tip.x - ring_tip.x) < 0.05 and abs(thumb_tip.y - ring_tip.y) < 0.05
    if thumb_ring_touch:
        return 'food'

    # Gesture 2: "I Love You" Gesture (Thumb, Index, and Pinky raised; Middle and Ring down)
    index_raised = index_tip.y < index_dip.y
    middle_raised = middle_tip.y < middle_dip.y
    ring_raised = ring_tip.y < ring_dip.y
    pinky_raised = pinky_tip.y < pinky_dip.y
    thumb_raised = thumb_tip.y < thumb_ip.y

    if thumb_raised and index_raised and pinky_raised and not middle_raised and not ring_raised:
        return 'I Love You'

    # Gesture 3: Coffee Gesture (Fist with Thumb extended)
    if not index_raised and not middle_raised and not ring_raised and not pinky_raised and thumb_raised:
        return 'Coffee Gesture'

    # Gesture 4: Tea Gesture (Palm facing up with fingers spread)
    fingers_spread = (index_raised and middle_raised and ring_raised and pinky_raised and not thumb_raised)
    if fingers_spread:
        return 'Tea Gesture'

    return None

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame color to RGB for Mediapipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to detect hands
    results = hands.process(rgb_frame)
    
    # Draw landmarks and detect gestures if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Detect the gesture
            gesture = detect_gesture(hand_landmarks)
            
            # Display the corresponding output
            if gesture:
                cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    
    # Show the frame
    cv2.imshow('Webcam', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
