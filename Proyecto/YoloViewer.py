# yolo_viewer.py
import tkinter as tk
from tkinter import ttk, font
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import torch
import numpy as np
import time
from threading import Thread
import datetime

# Configuración del modelo
MODEL_PATH = "best.torchscript"
DEVICE = "cpu"

# Paleta de colores profesional
COLORS = {
    "background": "#2c3e50",
    "primary": "#3498db",
    "secondary": "#2980b9",
    "accent": "#e74c3c",
    "text": "#ecf0f1",
    "success": "#2ecc71",
    "warning": "#f39c12"
}

class PremiumYoloViewer(tk.Tk):
    def __init__(self, model, device="cpu"):
        super().__init__()
        self.title("Sistema de Reconocimiento Facial - Proyecto Final PDI")
        self.geometry("1200x800")
        self.configure(bg=COLORS["background"])
        self.model = model
        self.device = device
        self.conf_thres = 0.5  # Valor fijo eliminando el slider
        self.cap = None
        self.running_camera = False
        self.current_frame = None
        self.last_detection = None
        self.access_granted = False

        # Configurar estilos premium
        self.setup_styles()
        self.create_widgets()
        self.bind_events()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configurar estilos para los widgets
        style.configure('TFrame', background=COLORS["background"])
        style.configure('TLabel', background=COLORS["background"], foreground=COLORS["text"])
        style.configure('TButton', font=('Helvetica', 12), padding=10)
        style.configure('Primary.TButton', background=COLORS["primary"], foreground='white')
        style.map('Primary.TButton',
                background=[('active', COLORS["secondary"]), ('disabled', '#7f8c8d')])
        style.configure('Accent.TButton', background=COLORS["accent"], foreground='white')
        style.configure('Success.TLabel', background=COLORS["background"], foreground=COLORS["success"], font=('Helvetica', 14, 'bold'))
        style.configure('Warning.TLabel', background=COLORS["background"], foreground=COLORS["warning"], font=('Helvetica', 14, 'bold'))
        style.configure('Title.TLabel', background=COLORS["background"], foreground=COLORS["text"], font=('Helvetica', 24, 'bold'))
        style.configure('Subtitle.TLabel', background=COLORS["background"], foreground=COLORS["text"], font=('Helvetica', 16))
        style.configure('TScale', background=COLORS["background"])

    def create_widgets(self):
        # Frame principal
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Header con título y logo
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        # Título y créditos (parte izquierda)
        text_frame = ttk.Frame(header_frame, style='Header.TFrame')
        text_frame.pack(side=tk.LEFT, fill=tk.Y, expand=True, padx=20)        
        
        # Título principal
        title_label = ttk.Label(text_frame, 
                              text="SISTEMA DE CONTROL DE ACCESO", 
                              style='Title.TLabel')
        title_label.pack(pady=(10, 0))

        # Subtítulo
        subtitle_label = ttk.Label(text_frame,
                                 text="Proyecto Final - PDI - TAM",
                                 style='Subtitle.TLabel')
        subtitle_label.pack()

        # Autores
        authors_label = ttk.Label(text_frame,
                                text="Juan Esteban Villada - Mateo Duque Gaviria - Sergio Andres Mora Orrego",
                                style='Authors.TLabel')
        authors_label.pack(pady=(0, 10))

        authors_label = ttk.Label(text_frame,
                                text="Ingeniería Electrónica",
                                style='Authors.TLabel')
        authors_label.pack(pady=(0, 10))       

        # Panel de control simplificado (sin slider)
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 20))

        # Botón de cámara
        self.cam_btn = ttk.Button(control_frame, 
                                 text="▶ INICIAR CÁMARA", 
                                 command=self.toggle_camera,
                                 style='Primary.TButton')
        self.cam_btn.pack(side=tk.LEFT, padx=5)

        # Botón para tomar foto
        self.capture_btn = ttk.Button(control_frame, 
                                    text="📸 CAPTURAR IMAGEN", 
                                    command=self.capture_image,
                                    style='Accent.TButton',
                                    state=tk.DISABLED)
        self.capture_btn.pack(side=tk.LEFT, padx=5)

        # Panel de visualización
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill=tk.BOTH, expand=True)

        # Vista de cámara
        self.camera_lbl = ttk.Label(display_frame, 
                                  text="Vista previa de cámara no activada",
                                  font=('Helvetica', 12),
                                  relief=tk.SUNKEN)
        self.camera_lbl.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Vista de resultados
        self.result_lbl = ttk.Label(display_frame, 
                                  text="Resultados de detección aparecerán aquí",
                                  font=('Helvetica', 12),
                                  relief=tk.SUNKEN)
        self.result_lbl.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Panel de estado
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(20, 0))

        # Información de detección
        self.detection_info = ttk.Label(status_frame,
                                      text="Estado: Esperando activación...",
                                      style='Subtitle.TLabel')
        self.detection_info.pack(side=tk.LEFT, padx=10)

        # Resultado de acceso
        self.access_label = ttk.Label(status_frame,
                                    text="ACCESO: NO VERIFICADO",
                                    style='Warning.TLabel')
        self.access_label.pack(side=tk.RIGHT, padx=10)

        # Configurar grid
        display_frame.columnconfigure(0, weight=1)
        display_frame.columnconfigure(1, weight=1)
        display_frame.rowconfigure(0, weight=1)

    def bind_events(self):
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def toggle_camera(self):
        if self.running_camera:
            self.stop_camera()
            self.cam_btn.config(text="▶ INICIAR CÁMARA")
            self.capture_btn.config(state=tk.DISABLED)
        else:
            self.start_camera()
            self.cam_btn.config(text="⏹ DETENER CÁMARA")
            self.capture_btn.config(state=tk.NORMAL)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.detection_info.config(text="Error: No se pudo abrir la cámara", style='Warning.TLabel')
            return
        
        self.running_camera = True
        self.detection_info.config(text="Cámara activada - Enfoque su rostro en la cámara", style='Success.TLabel')
        
        self.update_camera_view()

    def stop_camera(self):
        self.running_camera = False
        if self.cap:
            self.cap.release()
        self.camera_lbl.config(image='', text="Vista previa de cámara no activada")
        self.detection_info.config(text="Estado: Cámara desactivada", style='Subtitle.TLabel')
        self.access_label.config(text="ACCESO: NO VERIFICADO", style='Warning.TLabel')

    def update_camera_view(self):
        if self.running_camera:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = self.resize_to_fit(frame_rgb, max_height=500)
                
                img = Image.fromarray(frame_resized)
                self.tk_img = ImageTk.PhotoImage(image=img)
                
                self.camera_lbl.config(image=self.tk_img)
            
            self.after(30, self.update_camera_view)

    def capture_image(self):
        if self.current_frame is not None:
            self.last_detection = self.current_frame
            self.process_frame(self.current_frame)

    def process_frame(self, frame):
        # Procesar el frame actual
        tensor, orig_rgb = preprocess_image(frame)
        outputs = run_inference(self.model, tensor, self.device)
        
        detections_tensor = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        pred_bgr, counts = draw_detections(orig_rgb, detections_tensor, self.conf_thres)
        
        # Determinar acceso (aquí puedes personalizar la lógica)
        self.access_granted = len(counts) > 0 and max(counts.values()) > 0
        
        # Redimensionar para mostrar
        pred_resized = self.resize_to_fit(pred_bgr, max_height=500)
        
        # Convertir a PIL Image para añadir texto
        pred_pil = Image.fromarray(pred_resized)
        draw = ImageDraw.Draw(pred_pil)
        
        # Añadir marca de tiempo
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        font_small = ImageFont.load_default()
        draw.text((10, 10), f"Detectado: {timestamp}", fill=(255, 255, 255), font=font_small)
        
        # Añadir resultados de detección
        result_text = "DETECCIONES:\n"
        for cls_id, count in counts.items():
            result_text += f"Clase {cls_id}: {count} detecciones\n"
        
        draw.text((10, 30), result_text, fill=(255, 255, 255), font=font_small)
        
        # Añadir resultado de acceso
        access_text = "ACCESO PERMITIDO" if self.access_granted else "ACCESO DENEGADO"
        access_color = (0, 255, 0) if self.access_granted else (255, 0, 0)
        font_large = ImageFont.load_default()
        draw.text((pred_pil.width-200, 10), access_text, fill=access_color, font=font_large)
        
        # Convertir de vuelta a PhotoImage
        self.tk_result = ImageTk.PhotoImage(pred_pil)
        
        # Actualizar interfaz
        self.result_lbl.config(image=self.tk_result)
        self.access_label.config(
            text="ACCESO: PERMITIDO ✅" if self.access_granted else "ACCESO: DENEGADO ❌",
            style='Success.TLabel' if self.access_granted else 'Warning.TLabel'
        )
        
        # Actualizar información de detección
        if counts:
            self.detection_info.config(
                text=f"Detección exitosa - {len(counts)} clase(s) detectada(s)",
                style='Success.TLabel'
            )
        else:
            self.detection_info.config(
                text="No se detectaron objetos con el umbral actual",
                style='Warning.TLabel'
            )

    def resize_to_fit(self, image, max_height=500):
        h, w = image.shape[:2]
        scale = max_height / h
        new_size = (int(w * scale), max_height)
        return cv2.resize(image, new_size)

    def on_closing(self):
        self.stop_camera()
        self.destroy()

# Funciones del modelo (sin cambios)
def load_torchscript_model(model_path, device="cpu"):
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    print(f"Modelo cargado desde {model_path} en {device}")
    return model

def preprocess_image(image, input_size=(640, 640)):
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image.copy()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, input_size)
    img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return img_tensor, img_rgb

def run_inference(model, input_tensor, device="cpu"):
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        start = time.time()
        outputs = model(input_tensor)
        inference_time = (time.time() - start) * 1000
    print(f"Tiempo de inferencia: {inference_time:.2f} ms")
    return outputs

def draw_detections(img_bgr, detections_tensor, confidence_threshold=0.5):
    detections = detections_tensor[0] if len(detections_tensor.shape) == 3 else detections_tensor
    valid = detections[detections[:, 4] >= confidence_threshold]

    np.random.seed(42)
    colors = [(int(c[0]), int(c[1]), int(c[2])) for c in np.random.randint(0, 255, (100, 3))]

    out = img_bgr.copy()
    counts = {}
    for d in valid:
        x1, y1, x2, y2, conf, cls_id = d.tolist()
        x1, y1, x2, y2, cls_id = int(x1), int(y1), int(x2), int(y2), int(cls_id)
        color = colors[cls_id % len(colors)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"Clase {cls_id}: {conf:.2f}"
        cv2.putText(out, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        counts[cls_id] = counts.get(cls_id, 0) + 1
    return out, counts

if __name__ == "__main__":
    model = load_torchscript_model(MODEL_PATH, DEVICE)
    app = PremiumYoloViewer(model, DEVICE)
    app.mainloop()