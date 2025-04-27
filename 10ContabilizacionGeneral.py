import cv2
from ultralytics import YOLO

modelo = YOLO("yolov8l.pt")
# modelo.to("cuda")  # Descomenta si quieres usar GPU

cap = cv2.VideoCapture("./videos/personas.mp4")
cv2.namedWindow("Resultado", cv2.WINDOW_NORMAL)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    # Verificar que el frame no esté vacío
    if frame is None or frame.size == 0:
        continue

    try:
        resultados = modelo.track(frame, persist=True, conf=0.5)

        cuenta = 0  # Reiniciar contador en cada frame
        if resultados and resultados[0].boxes is not None:
            cajas = resultados[0].boxes
            for caja in cajas:
                cajaCoords = caja.xyxy[0].cpu().numpy()  # Asegura que esté en numpy
                centroCaja = (int((cajaCoords[0] + cajaCoords[2]) / 2),
                              int((cajaCoords[1] + cajaCoords[3]) / 2))
                cv2.circle(frame, centroCaja, 5, (0, 0, 255), -1)
                cuenta += 1

        # Escribir el número de personas detectadas
        cadena = f"{cuenta} personas"
        tamanoTexto, _ = cv2.getTextSize(cadena, cv2.FONT_HERSHEY_TRIPLEX, 5, 2)
        cv2.putText(frame, cadena, (0, tamanoTexto[1] + 15),
                    cv2.FONT_HERSHEY_TRIPLEX, 5, (100, 0, 100), 5)

    except Exception as e:
        print(f"Error en el frame: {e}")

    cv2.imshow("Resultado", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
