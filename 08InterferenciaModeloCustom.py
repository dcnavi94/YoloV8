from ultralytics import YOLO
import cv2

modelo = YOLO("placas.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow("Resultados", cv2.WINDOW_NORMAL)

while True:
    capturaOK, frame = cap.read()
    if not capturaOK:
        break
    
    resultados = modelo(frame, stream=True, verbose=False, conf=0.5)
    
    for r in resultados:
        cajas = r.boxes
        
        for caja in cajas:
            porcentajeCoincidencia = int(caja.conf[0]*100)
            clase = int(caja.cls[0])
            # print(nombresDeClase[clase], porcentajeCoincidencia)
            
            x1, y1, x2, y2 = caja.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 3)
            cv2.putText(frame, r.names[clase], (x1,y1+50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
    cv2.imshow("Resultados", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()