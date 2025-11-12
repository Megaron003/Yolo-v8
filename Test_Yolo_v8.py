from ultralytics import YOLO
import cv2

# Carregar modelo
model = YOLO('yolov8n.pt')

# Inicializar webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Realizar detecção
    results = model(frame, verbose=False)
    
    # Plotar resultados no frame
    annotated_frame = results[0].plot()
    
    # Mostrar frame
    cv2.imshow('YOLOv8 Webcam', annotated_frame)
    
    # Sair com 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"System ready!")

cap.release()
cv2.destroyAllWindows()