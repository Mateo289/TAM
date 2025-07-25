🚀 Proyecto Final: Sistema de Control de Acceso por Reconocimiento Facial con YOLOv10
¡Bienvenido al repositorio del proyecto final del curso Teoría de Aprendizaje de Máquina! Aquí encontrarás todo el código, modelos y documentación necesarios para entender y replicar nuestro sistema de control de acceso basado en reconocimiento facial, optimizado para dispositivos embebidos como la NVIDIA Jetson Nano.

📌 Resumen del Proyecto
Este proyecto implementa un sistema de control de acceso automatizado y seguro utilizando técnicas avanzadas de Deep Learning (YOLOv10) para el reconocimiento facial en tiempo real. El sistema es capaz de:

Clasificar rostros en dos categorías: "Acceso Permitido" y "Acceso Denegado".

Operar en tiempo real (inferencia en 380 ms) en hardware limitado (Jetson Nano).

Alcanzar una precisión del 92.1% en pruebas de laboratorio.

Ser económico y escalable, gracias al uso de herramientas de código abierto.

🛠️ Tecnologías y Métodos Clave
Componente	Detalle
Modelo Base	YOLOv10s (optimizado para edge computing)
Dataset	933 imágenes etiquetadas (Roboflow)
Optimización	Conversión a TorchScript (modelo final de solo 14 MB)
Hardware	Entrenamiento: GPU NVIDIA T4 (Kaggle) / Despliegue: Jetson Nano (4GB)
Interfaz	GUI en Python con Tkinter

🔥 ¿Cómo lo hicimos?
Dataset y Preprocesamiento:

Usamos Roboflow para gestionar y versionar el dataset de rostros.

Preprocesamiento automático en YOLO: redimensionamiento a 640x640 y normalización.

Entrenamiento del Modelo:

Fine-tuning de YOLOv10s (100 épocas, batch size=16) en Kaggle (GPU Tesla T4).

Resultado: best.pt con 92.1% de precisión.

Optimización para Jetson Nano:

Exportamos el modelo a TorchScript (best.torchscript) para reducir latencia y dependencias.

Validación y GUI:

Scripts de inferencia con OpenCV y medición de tiempos.

Interfaz gráfica para pruebas locales (Tkinter).

🎯 Resultados Destacados
Métrica	Valor Obtenido	Objetivo
Precisión	92.1%	>85% ✅
Tiempo de Inferencia	380 ms	<500 ms ✅
Tamaño del Modelo	14 MB	<20 MB ✅
💡 Desafíos y Trabajo Futuro
Limitaciones: Sensibilidad a cambios de iluminación y distorsión de aspecto.

Mejoras Propuestas:

Migrar a TensorRT para inferencia más rápida.

Añadir módulo anti-spoofing contra ataques con fotos/videos.

Ampliar el dataset con condiciones de iluminación diversas.

👨‍💻 Autores
Juan Esteban Villada Sierra

Sergio Andrés Mora Orrego

Mateo Duque Gaviria

*Universidad Nacional de Colombia - Sede Manizales, 2025-1*
