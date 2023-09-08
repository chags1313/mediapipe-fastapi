from fastapi import FastAPI, File, Form, HTTPException
from fastapi.responses import JSONResponse
import mediapipe as mp
import numpy as np
import cv2

app = FastAPI()

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

white_landmark_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
white_connection_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2)
green_landmark_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
green_connection_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

@app.post("/process_image")
async def process_image(image: bytes = File(...), seen_all_keypoints: bool = Form(False)):
    try:
        image_np = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        results = holistic.process(image)

        keypoints = []
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                keypoints.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })

        all_visible = all([landmark.visibility > 0.5 for landmark in results.pose_landmarks.landmark])

        if not seen_all_keypoints:
            if results.pose_landmarks:
                if all_visible:
                    drawing_spec = green_landmark_drawing_spec
                    connection_spec = green_connection_drawing_spec
                else:
                    drawing_spec = white_landmark_drawing_spec
                    connection_spec = white_connection_drawing_spec

                mp.solutions.drawing_utils.draw_landmarks(
                    image, 
                    results.pose_landmarks, 
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=connection_spec
                )

        _, img_encoded = cv2.imencode('.jpg', image)
        img_bytes = img_encoded.tobytes()

        return {
            'image': img_bytes.decode('latin1'),
            'keypoints': keypoints,
            'all_keypoints_visible': all_visible
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
