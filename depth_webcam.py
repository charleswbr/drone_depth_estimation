import argparse
import cv2
import matplotlib
import numpy as np
import torch
import time
import threading
import queue
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import io
from PIL import Image

from metric_depth.depth_anything_v2.dpt import DepthAnythingV2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 avec Webcam')
    
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--batch-size', type=int, default=1, help='Taille du batch pour le traitement')
    parser.add_argument('--mixed-precision', action='store_true', help='Utiliser la précision mixte pour accélérer le calcul sur GPU')
    parser.add_argument('--camera-id', type=int, default=0, help='ID de la caméra à utiliser')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='afficher uniquement la prédiction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='ne pas appliquer de palette colorée')
    parser.add_argument('--save-video', dest='save_video', action='store_true', help='enregistrer la sortie en vidéo')
    parser.add_argument('--output-path', type=str, default='depth_webcam.mp4', help='chemin du fichier de sortie vidéo')
    
    args = parser.parse_args()

    dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
    max_depth = 20 # 20 for indoor model, 80 for outdoor model
    
    # Vérifier la disponibilité du GPU ou MPS
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Utilisation du périphérique: {DEVICE}")
    
    # Optimisations pour CUDA
    if DEVICE == 'cuda':
        # Permettre l'utilisation de TensorFloat32 pour une inférence plus rapide sur les GPU Ampere+
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print("Optimisations CUDA activées")
    
    # Configuration du modèle selon l'encodeur choisi
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    # Charger le modèle
    print(f"Chargement du modèle Depth Anything V2 avec l'encodeur {args.encoder}...")
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': max_depth})
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_{dataset}_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    # Initialiser la webcam
    print(f"Initialisation de la webcam (ID: {args.camera_id})...")
    cap = cv2.VideoCapture(args.camera_id)
    
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la webcam")
        exit(1)
    
    # Obtenir les dimensions de la webcam
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Configuration pour l'affichage
    margin_width = 50
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    # Configuration pour l'enregistrement vidéo
    if args.save_video:
        if args.pred_only:
            output_width = frame_width
        else:
            output_width = frame_width * 2 + margin_width
            
        out = cv2.VideoWriter(args.output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))
    
    # Créer une file d'attente pour le traitement parallèle
    frame_queue = queue.Queue(maxsize=3)  # Limiter à 3 frames pour éviter la consommation excessive de mémoire
    result_queue = queue.Queue()

    # Fonction pour le thread d'inférence
    def inference_worker():
        while True:
            frame = frame_queue.get()
            if frame is None:  # Signal de fin
                frame_queue.task_done()
                break
                
            # Utiliser la précision mixte si demandé
            if args.mixed_precision and DEVICE == 'cuda':
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        raw_depth = depth_anything.infer_image(frame, args.input_size)
            else:
                with torch.no_grad():
                    raw_depth = depth_anything.infer_image(frame, args.input_size)
                    print(raw_depth)
            
            # Store raw depth values (in meters based on the model's training)
            # The model is calibrated for max_depth (20m indoor or 80m outdoor)
            
            # Normalisation et préparation de l'image de profondeur
            normalized_depth = (raw_depth - raw_depth.min()) / (raw_depth.max() - raw_depth.min()) * 255.0
            normalized_depth = normalized_depth.astype(np.uint8)
            
            # Mettre le résultat dans la file d'attente
            result_queue.put((frame, normalized_depth, raw_depth))
            frame_queue.task_done()

    # Démarrer le thread d'inférence
    inference_thread = threading.Thread(target=inference_worker, daemon=True)
    inference_thread.start()

    try:
        print("Traitement en cours... Appuyez sur 'q' pour quitter.")
        last_time = time.time()
        frame_count = 0
        running_fps = 0
        
        # Première image pour initialiser
        ret, frame = cap.read()
        if ret:
            frame_queue.put(frame)
        
        while True:
            # Lire une nouvelle image en attendant le résultat de la précédente
            ret, new_frame = cap.read()
            if not ret:
                print("Erreur: Impossible de lire depuis la webcam")
                break
                
            # Mettre en file d'attente si possible
            if not frame_queue.full():
                frame_queue.put(new_frame)
            
            # Vérifier si un résultat est disponible
            if not result_queue.empty():
                frame, normalized_depth, raw_depth = result_queue.get()
                
                # Create a colorbar figure with matplotlib
                fig = plt.figure(figsize=(1.5, 4))
                ax = plt.axes()
                norm = Normalize(vmin=0, vmax=max_depth)
                sm = ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax)
                cbar.set_label('Distance (m)', rotation=270, labelpad=15)
                ax.remove()  # Hide the axes, show only colorbar
                
                # Convert matplotlib figure to OpenCV image
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
                buf.seek(0)
                legend_img = np.array(Image.open(buf))
                legend_img = cv2.cvtColor(legend_img, cv2.COLOR_RGB2BGR)
                legend_height = frame_height
                legend_img = cv2.resize(legend_img, (100, legend_height))
                plt.close(fig)
                
                # Appliquer la colorisation si nécessaire
                if args.grayscale:
                    depth_display = np.repeat(normalized_depth[..., np.newaxis], 3, axis=-1)
                else:
                    depth_display = (cmap(normalized_depth/255.0)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
                
                # Extract actual distance at center point (in meters)
                center_y, center_x = raw_depth.shape[0] // 2, raw_depth.shape[1] // 2
                center_depth = raw_depth[center_y, center_x]
                
                # Add distance measurement points
                cv2.circle(depth_display, (center_x, center_y), 5, (0, 0, 255), -1)
                cv2.putText(depth_display, f"{center_depth:.2f}m", 
                            (center_x + 10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 0, 255), 2)
                
                # Préparer l'affichage avec légende
                if args.pred_only:
                    display_frame = cv2.hconcat([depth_display, legend_img])
                else:
                    split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
                    display_frame = cv2.hconcat([frame, split_region, depth_display, legend_img])
                
                # Enregistrer la vidéo si demandé
                if args.save_video:
                    out.write(display_frame)
                
                # Calculer et afficher le FPS
                frame_count += 1
                if frame_count % 10 == 0:  # Mettre à jour le FPS tous les 10 frames
                    current_time = time.time()
                    running_fps = 10 / (current_time - last_time)
                    last_time = current_time
                
                cv2.putText(display_frame, f"FPS: {running_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Afficher le résultat
                cv2.imshow('Depth Anything V2 - Webcam', display_frame)
            
            # Vérifier si l'utilisateur veut quitter
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # Petite pause pour éviter de saturer le CPU
            time.sleep(0.001)
    
    finally:
        # Arrêter proprement le thread d'inférence
        if inference_thread.is_alive():
            frame_queue.put(None)  # Signal pour arrêter le thread
            inference_thread.join(timeout=1.0)
            
        # Vider les files d'attente
        while not frame_queue.empty():
            try:
                frame_queue.get_nowait()
                frame_queue.task_done()
            except queue.Empty:
                break
                
        while not result_queue.empty():
            try:
                result_queue.get_nowait()
            except queue.Empty:
                break
        
        # Libérer la mémoire CUDA si nécessaire
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
            
        # Nettoyage
        cap.release()
        if args.save_video:
            out.release()
        cv2.destroyAllWindows()
        print("Programme terminé.")