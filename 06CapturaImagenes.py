import cv2
import uuid

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    cv2.imshow("Imagen", frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):
        miUUID = str(uuid.uuid1())
        cv2.imwrite("./imagenes/"+miUUID+".jpg", frame)
        
cap.release()
cv2.destroyAllWindows()