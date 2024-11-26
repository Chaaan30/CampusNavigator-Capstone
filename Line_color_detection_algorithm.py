import cv2
import numpy as np
import serial
import time

# Define color ranges in HSV format
COLOR_RANGES = {
    "red": ((0, 120, 70), (10, 255, 255)),
    "green": ((40, 50, 50), (90, 255, 255)),
    "blue": ((100, 150, 0), (140, 255, 255)),
    "yellow": ((20, 100, 100), (30, 255, 255)),
}

# Messages to send for each detected color
FLOOR_MESSAGES = {
    "red": "first floor",
    "green": "second floor",
    "blue": "third floor",
    "yellow": "fourth floor",
}

# Initialize serial communication with Arduino
# Replace 'COM3' with the correct port for your Arduino
arduino = serial.Serial(port='COM5', baudrate=9600, timeout=1)
time.sleep(2)  # Give some time for the connection to establish

def send_to_arduino(message):
    """Send a message to the Arduino via serial and print the response."""
    arduino.write((message + '\n').encode())
    print(f"Sent to Arduino: {message}")
    time.sleep(0.1)  # Small delay to ensure Arduino has time to respond
    if arduino.in_waiting > 0:
        response = arduino.readline().decode().strip()
        print(f"Arduino Response: {response}")

def detect_lines_and_send(frame, color_ranges):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    result_frame = frame.copy()
    
    for color_name, (lower, upper) in color_ranges.items():
        # Create masks for the specified color range
        mask = cv2.inRange(hsv_frame, np.array(lower), np.array(upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        
        # Detect edges and contours
        edges = cv2.Canny(mask, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process detected contours
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(result_frame, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Send the floor message to Arduino
                if color_name in FLOOR_MESSAGES:
                    send_to_arduino(FLOOR_MESSAGES[color_name])
    
    return result_frame

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Detect lines and send messages
        result_frame = detect_lines_and_send(frame, COLOR_RANGES)
        
        # Show the result
        cv2.imshow("Color Line Detection", result_frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    arduino.close()

if __name__ == "__main__":
    main()
