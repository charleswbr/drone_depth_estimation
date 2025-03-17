import argparse
import cv2
import matplotlib
import numpy as np
import time
import threading
import queue
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import io
from PIL import Image
import onnxruntime
import torch  # Only used here to check for CUDA availability (optional)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 avec Webcam (ONNX Quantized Model)')
    
    parser.add_argument('--input-size', type=int, default=224)
    # The encoder argument and model_configs are no longer needed when using the ONNX model.
    parser.add_argument('--batch-size', type=int, default=1, help='Taille du batch pour le traitement')
    parser.add_argument('--camera-id', type=int, default=0, help='ID de la caméra à utiliser')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='afficher uniquement la prédiction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='ne pas appliquer de palette colorée')
    parser.add_argument('--save-video', dest='save_video', action='store_true', help='enregistrer la sortie en vidéo')
    parser.add_argument('--output-path', type=str, default='depth_webcam.mp4', help='chemin du fichier de sortie vidéo')
    
    args = parser.parse_args()

    # Model parameters (update these if your model was calibrated on a different max_depth)
    max_depth = 20  # 20 for indoor model, 80 for outdoor model

    # Set up the ONNX Runtime provider
    providers = ['CPUExecutionProvider']
    if torch.cuda.is_available():
        providers = ['CUDAExecutionProvider']
    print(f"Chargement du modèle ONNX quantized INT8 avec providers: {providers}...")
    
    # Create the ONNX Runtime session (update the model path as needed)
    ort_session = onnxruntime.InferenceSession("checkpoints/depth_anything_v2.onnx", providers=providers)
    
    # Initialize webcam
    print(f"Initialisation de la webcam (ID: {args.camera_id})...")
    cap = cv2.VideoCapture(args.camera_id)
    
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la webcam")
        exit(1)
    
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
    
    # Preprocessing and inference function for ONNX quantized model
    def infer_image(frame, input_size):
        # Convert BGR to RGB and resize to the expected input dimensions
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (input_size, input_size))
        
        # Change layout from HWC to CHW and add batch dimension
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        
        # Get model input information
        input_info = ort_session.get_inputs()[0]
        input_type = input_info.type
        
        # Depending on your quantized model's expected input type, apply the appropriate conversion.
        if input_type == 'tensor(int8)':
            # Example quantization parameters (update these with the actual values for your model)
            scale = 0.1
            zero_point = 128
            # Quantize the image: assume original image is in [0, 255]
            image = np.clip(image / scale + zero_point, 0, 255).astype(np.int8)
        else:
            # If the model expects float input, normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
        
        # Run inference with the ONNX model
        output_name = ort_session.get_outputs()[0].name
        outputs = ort_session.run([output_name], {input_info.name: image})
        raw_depth = outputs[0]
        # Remove the batch dimension if present
        if raw_depth.shape[0] == 1:
            raw_depth = raw_depth[0]
        return raw_depth

    # Create queues for parallel processing
    frame_queue = queue.Queue(maxsize=3)  # Limit to avoid excessive memory use
    result_queue = queue.Queue()

    # Worker thread for inference
    def inference_worker():
        while True:
            frame = frame_queue.get()
            if frame is None:  # Stop signal
                frame_queue.task_done()
                break
                
            # Run inference using the ONNX model
            raw_depth = infer_image(frame, args.input_size)
            
            # Normalize the raw depth for display
            normalized_depth = (raw_depth - raw_depth.min()) / (raw_depth.max() - raw_depth.min()) * 255.0
            normalized_depth = normalized_depth.astype(np.uint8)
            
            result_queue.put((frame, normalized_depth, raw_depth))
            frame_queue.task_done()

    # Start the inference thread
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
            ret, new_frame = cap.read()
            if not ret:
                print("Erreur: Impossible de lire depuis la webcam")
                break
                
            if not frame_queue.full():
                frame_queue.put(new_frame)
            
            if not result_queue.empty():
                frame, normalized_depth, raw_depth = result_queue.get()
                
                # Create a colorbar using matplotlib
                fig = plt.figure(figsize=(1.5, 4))
                ax = plt.axes()
                norm = Normalize(vmin=0, vmax=max_depth)
                sm = ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax)
                cbar.set_label('Distance (m)', rotation=270, labelpad=15)
                ax.remove()  # Only show the colorbar
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
                buf.seek(0)
                legend_img = np.array(Image.open(buf))
                legend_img = cv2.cvtColor(legend_img, cv2.COLOR_RGB2BGR)
                
                # Resize legend_img to match the height of the frame
                legend_img = cv2.resize(legend_img, (100, frame.shape[0]))
                
                # Apply colorization if needed
                if args.grayscale:
                    depth_display = np.repeat(normalized_depth[..., np.newaxis], 3, axis=-1)
                else:
                    # Resize normalized_depth to match the frame size before applying the colormap
                    normalized_depth_resized = cv2.resize(normalized_depth, (frame.shape[1], frame.shape[0]))
                    depth_display = (cmap(normalized_depth_resized/255.0)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
                
                # Mark the center depth value on the depth image
                # Adjust coordinates to match the resized depth_display
                center_y, center_x = depth_display.shape[0] // 2, depth_display.shape[1] // 2
                
                # Get the depth value from original raw_depth, but scale coordinates appropriately
                orig_y = int(center_y * raw_depth.shape[0] / depth_display.shape[0])
                orig_x = int(center_x * raw_depth.shape[1] / depth_display.shape[1])
                center_depth = raw_depth[orig_y, orig_x]
                
                # Scale the depth value to meters using max_depth
                # raw_depth values are typically normalized between 0-1, so scale to real-world distance
                real_depth_meters = center_depth * max_depth / raw_depth.max()
                
                cv2.circle(depth_display, (center_x, center_y), 5, (0, 0, 255), -1)
                cv2.putText(depth_display, f"{real_depth_meters:.2f}m", 
                            (center_x + 10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 0, 255), 2)
                
                # Prepare display frame with legend
                split_region = np.ones((frame.shape[0], margin_width, 3), dtype=np.uint8) * 255
                
                # Ensure all arrays have the same height, number of channels, and data type
                if args.pred_only:
                    # Make sure depth_display and legend_img have the same height
                    if depth_display.shape[0] != legend_img.shape[0]:
                        legend_img = cv2.resize(legend_img, (legend_img.shape[1], depth_display.shape[0]))
                    display_frame = cv2.hconcat([depth_display, legend_img])
                else:
                    # Ensure all components have the same height
                    height = frame.shape[0]
                    if depth_display.shape[0] != height:
                        depth_display = cv2.resize(depth_display, (depth_display.shape[1], height))
                    if legend_img.shape[0] != height:
                        legend_img = cv2.resize(legend_img, (legend_img.shape[1], height))
                    if split_region.shape[0] != height:
                        split_region = np.ones((height, margin_width, 3), dtype=np.uint8) * 255
                        
                    # Ensure all components have the same type
                    frame = frame.astype(np.uint8)
                    depth_display = depth_display.astype(np.uint8)
                    legend_img = legend_img.astype(np.uint8)
                    split_region = split_region.astype(np.uint8)
                    
                    # Now concatenate
                    display_frame = cv2.hconcat([frame, split_region, depth_display, legend_img])
                
                if args.save_video:
                    out.write(display_frame)
                
                frame_count += 1
                if frame_count % 10 == 0:
                    current_time = time.time()
                    running_fps = 10 / (current_time - last_time)
                    last_time = current_time
                
                cv2.putText(display_frame, f"FPS: {running_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Depth Anything V2 - Webcam', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            time.sleep(0.001)
    
    finally:
        if inference_thread.is_alive():
            frame_queue.put(None)  # Signal the thread to stop
            inference_thread.join(timeout=1.0)
            
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
        
        # Clean up resources
        cap.release()
        if args.save_video:
            out.release()
        cv2.destroyAllWindows()
        print("Programme terminé.")
