import cv2
import os
import pandas as pd
from deepface import DeepFace
from datetime import datetime

cap = cv2.VideoCapture(0)

if not os.path.exists("attendance.csv"):
    pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv("attendance.csv", index=False)

print("Look at camera. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    cv2.imshow("Authenticate", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    for file in os.listdir("registered_faces"):
        name = file.split(".")[0]
        ref_img = f"registered_faces/{file}"

        try:
            result = DeepFace.verify(
                frame,
                ref_img,
                model_name="Facenet",
                enforce_detection=True
            )

            if result["verified"]:
                now = datetime.now()
                df = pd.read_csv("attendance.csv")

                if not ((df["Name"] == name) & (df["Date"] == now.strftime("%Y-%m-%d"))).any():
                    df.loc[len(df)] = [name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")]
                    df.to_csv("attendance.csv", index=False)
                    print(f"{name} marked present")

        except:
            pass

cap.release()
cv2.destroyAllWindows()
