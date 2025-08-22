import cv2
import numpy as np

def generate_multiple_markers():
    # ArUco dictionary define करें
    aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    
    # Multiple marker IDs जो आप generate करना चाहते हैं
    marker_ids = [1, 2, 3, 4, 5]
    
    # Marker size (pixels में)
    marker_size = 300
    
    print("Generating ArUco markers...")
    
    for marker_id in marker_ids:
        # Blank image create करें marker के लिए
        marker_image = np.zeros((marker_size, marker_size, 1), dtype="uint8")
        
        # ArUco marker generate करें
        cv2.aruco.generateImageMarker(
            aruco_dictionary,    # Dictionary
            marker_id,          # Marker ID
            marker_size,        # Size in pixels
            marker_image,       # Output image
            1                   # Border bits
        )
        
        # Filename create करें
        filename = f"marker_{marker_id}.png"
        
        # Marker save करें
        cv2.imwrite(filename, marker_image)
        print(f"Generated: {filename}")
        
        # Optional: हर marker को display करना चाहते हैं तो uncomment करें
        # cv2.imshow(f"ArUco Marker ID: {marker_id}", marker_image)
        # cv2.waitKey(1000)  # 1 second display
    
    print("All markers generated successfully!")
    # cv2.destroyAllWindows()  # अगर display किया हो तो uncomment करें

# Function को call करें
if __name__ == "__main__":
    generate_multiple_markers()
