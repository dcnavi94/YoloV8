import cv2
from ultralytics import YOLO

# Cargar el modelo YOLOv8
model = YOLO("yolov8n.pt")

# Abrir cámara (0 = cámara por defecto)
cap = cv2.VideoCapture(0)

# Definir zona segura (x1, y1, x2, y2)
zona_vigilada = (200, 100, 450, 400)  # puedes modificarla a gusto

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Ejecutar predicción
    results = model(frame, verbose=False)[0]

    # Dibujar la zona de vigilancia
    x1, y1, x2, y2 = zona_vigilada
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(frame, "ZONA VIGILADA", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Revisar objetos detectados
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if model.names[cls] == "person":
            # Coordenadas del objeto detectado
            xA, yA, xB, yB = map(int, box.xyxy[0])

            # Dibujar persona
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
            cv2.putText(frame, "Persona", (xA, yA - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Centro de la caja
            centro_x = (xA + xB) // 2
            centro_y = (yA + yB) // 2

            # Verificar si está dentro del área vigilada
            if x1 < centro_x < x2 and y1 < centro_y < y2:
                print("Intruso")

    # Mostrar el resultado en una ventana
    cv2.imshow("Deteccion de Intrusos", frame)

    # Salir con la tecla ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
