import cv2
from ultralytics import YOLO

modelo = YOLO("yolov8m.pt")
#modelo.to("cuda")
clases = modelo.model.names
clasesConteo = {"person":0, "bicycle":0, "dog":0}

cap = cv2.VideoCapture("./videos/combinado2.mp4")
cv2.namedWindow("Resultado", cv2.WINDOW_NORMAL)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    
    resultados = modelo.track(frame, persist=True, classes=[0,1,16], conf=0.9)
    
    cajas = resultados[0].boxes
    for caja in cajas:
        cajaCoords = caja.xyxy[0]
        centroCaja = (int((cajaCoords[0] + cajaCoords[2]) /2), 
                      int((cajaCoords[1] + cajaCoords[3]) /2))
        cv2.circle(frame, centroCaja, 5, (0,0,255), -1)
        clase = int(caja.cls[0])
        clasesConteo[clases[clase]] += 1
    
    cadena1 = str(clasesConteo["person"]) + " personas"
    cadena2 = str(clasesConteo["bicycle"]) + " bicicletas"
    cadena3 = str(clasesConteo["dog"]) + " perros"
    tamanoTexto, _ = cv2.getTextSize(cadena1, cv2.FONT_HERSHEY_TRIPLEX, 5,2)
    cv2.putText(frame, cadena1, (0, tamanoTexto[1]+15), cv2.FONT_HERSHEY_TRIPLEX,
                5, (100,0,100), 5)
    cv2.putText(frame, cadena2, (0, tamanoTexto[1]*2+60), cv2.FONT_HERSHEY_TRIPLEX,
                5, (100,0,100), 5)
    cv2.putText(frame, cadena3, (0, tamanoTexto[1]*3+105), cv2.FONT_HERSHEY_TRIPLEX,
                5, (100,0,100), 5)
    
    cv2.imshow("Resultado", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    clasesConteo = {"person":0, "bicycle":0, "dog":0}
    
cap.release()
cv2.destroyAllWindows()