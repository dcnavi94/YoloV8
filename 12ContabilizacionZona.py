import cv2
from ultralytics import YOLO
import numpy as np

modelo = YOLO("yolov8x.pt")
# modelo.to("cuda")

areaConteo = np.array([(1000,250), (1500,250), (1300,1000), (600,1000)])

cap = cv2.VideoCapture("./videos/personas.mp4")
cv2.namedWindow("Resultado", cv2.WINDOW_NORMAL)

cuenta = 0
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    
    #bytetrack.yaml
    #botsort.yaml
    
    resultados = modelo.track(frame, persist=True, classes=0, conf=0.9, tracker="bytetrack.yaml")
    
    cajas = resultados[0].boxes
    for caja in cajas:
        cajaCoords = caja.xyxy[0]
        centroCaja = (int((cajaCoords[0] + cajaCoords[2]) /2), 
                      int((cajaCoords[1] + cajaCoords[3]) /2))
        cv2.circle(frame, centroCaja, 5, (0,0,255), -1)
        res = cv2.pointPolygonTest(areaConteo, centroCaja, False)
        if res == 1:
            cuenta += 1
    
    cadena = str(cuenta) + " personas"
    tamanoTexto, _ = cv2.getTextSize(cadena, cv2.FONT_HERSHEY_TRIPLEX, 5,2)
    cv2.putText(frame, cadena, (0, tamanoTexto[1]+15), cv2.FONT_HERSHEY_TRIPLEX,
                5, (100,0,100), 5)
    
    puntosPoly = areaConteo.reshape((-1,1,2))
    cv2.polylines(frame, [puntosPoly], isClosed=True, color=(0,255,0), thickness=5)
    
    
    cv2.imshow("Resultado", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    cuenta = 0
    
cap.release()
cv2.destroyAllWindows()