#En este código utilizamos r.names para obtener los nombres de clase
#A su vez puedes eliminar el ciclo for r in resultados:  y solo utilizar r[0], esto al tratarse de una sola imagen.
from ultralytics import YOLO
import cv2

modelo = YOLO("yolov8x.pt")
img = cv2.imread("1.jpg")
cv2.namedWindow("Resultados", cv2.WINDOW_NORMAL)

resultados = modelo(img, stream=False, verbose=False)

for r in resultados:
    cajas = r.boxes
    
    for caja in cajas:
        porcentajeCoincidencia = int(caja.conf[0]*100)
        clase = int(caja.cls[0])
        
        x1, y1, x2, y2 = caja.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 3)
        cv2.putText(img, r.names[clase], (x1,y1+50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
        
cv2.imshow("Resultados", img)
cv2.waitKey()
cv2.destroyAllWindows()