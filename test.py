#!/Users/irialmckeown/Desktop/peoplecounteryolov8-main/myvenv/bin/python
#versione windows
import cv2
import numpy as np
from ultralytics import YOLO

def get_effect_parameters():
    """Richiede all'utente i parametri degli effetti con validazione"""
    print("=== IMPOSTAZIONI EFFETTI ===")
    
    while True:
        try:
            noise_sigma = float(input("Intensità rumore gaussiano (0-100, consigliato 25): "))
            if not 0 <= noise_sigma <= 100:
                raise ValueError
            break
        except ValueError:
            print("Inserire un numero tra 0 e 100")

    while True:
        try:
            opacity_alpha = float(input("Opacità (0.0-1.0, 1=nessuna opacità): "))
            if not 0.0 <= opacity_alpha <= 1.0:
                raise ValueError
            break
        except ValueError:
            print("Inserire un numero tra 0.0 e 1.0")

    while True:
        try:
            quant_levels = int(input("Livelli di quantizzazione (2-256, 256=nessuna): "))
            if not 2 <= quant_levels <= 256:
                raise ValueError
            break
        except ValueError:
            print("Inserire un intero tra 2 e 256")

    return noise_sigma, opacity_alpha, quant_levels

def add_gaussian_noise(image, mean=0, sigma=25):
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_opacity(image, alpha=0.7):
    overlay = image.copy()
    output = image.copy()
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output

def quantize_image(image, levels=4):
    step = 255 / (levels - 1)
    quantized = np.round(image / step) * step
    return quantized.astype(np.uint8)

# Inizializzazione
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

# Richiesta parametri all'utente
NOISE_SIGMA, OPACITY_ALPHA, QUANT_LEVELS = get_effect_parameters()

people_count = 0
previous_people = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Pipeline di elaborazione
    noisy_frame = add_gaussian_noise(frame, sigma=NOISE_SIGMA)
    quantized_frame = quantize_image(noisy_frame, levels=QUANT_LEVELS)
    processed_frame = add_opacity(quantized_frame, alpha=OPACITY_ALPHA)
    
    frame_resized = cv2.resize(processed_frame, (800, 800))
    frame_for_detection = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    
    current_people = set()
    results = model(frame_for_detection, verbose=False)
    
    for i, box in enumerate(results[0].boxes):
        if box.cls == 0:  # Solo persone
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            current_people.add(i)
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_resized, f"ID: {i}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    # Aggiornamento conteggio
    new_people = current_people - previous_people
    exited_people = previous_people - current_people
    people_count += len(new_people) - len(exited_people)
    previous_people = current_people.copy()

    # Definizione testo e stili
    params_text = [
        f"TOTALE: {people_count}",
        f"Rumore: {NOISE_SIGMA}/100",
        f"Opacita: {OPACITY_ALPHA:.1f}/1.0",
        f"Livelli colore: {QUANT_LEVELS}/256"
    ]
    
    # Configurazione stili per ogni riga (colore, scala, spessore)
    text_styles = [
        ((0, 255, 0), 1.2, 3),    # Verde brillante, grande, spesso (TOTALE)
        ((255, 150, 50), 0.8, 2),  # Arancione
        ((50, 150, 255), 0.8, 2),  # Blu chiaro
        ((255, 50, 150), 0.8, 2)   # Rosa
    ]
    
    # Disegna ogni riga con il suo stile
    for i, (text, (color, scale, thickness)) in enumerate(zip(params_text, text_styles)):
        # Ombreggiatura per migliorare la leggibilità
        cv2.putText(
            frame_resized, text, (11, 30 + i*40),
            cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+1, cv2.LINE_AA
        )
        # Testo principale
        cv2.putText(
            frame_resized, text, (10, 30 + i*40),
            cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA
        )
    
    # Istruzioni per l'utente
    cv2.putText(
        frame_resized, "ESC: Esci | R: Ripristina impostazioni", (10, frame_resized.shape[0] - 10),
        cv2.FONT_HERSHEY_PLAIN, 1, (200,200,200), 1, cv2.LINE_AA
    )
    
    cv2.imshow("People Counter with Effects", frame_resized)
    
    key = cv2.waitKey(1)
    if key == 27:  # ESC per uscire
        break
    elif key == ord('r'):  # R per ripristinare impostazioni
        NOISE_SIGMA, OPACITY_ALPHA, QUANT_LEVELS = get_effect_parameters()

cap.release()
cv2.destroyAllWindows()