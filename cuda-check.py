import torch
import sys
import subprocess

def get_nvidia_smi_output():
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, text=True)
        return result.stdout
    except FileNotFoundError:
        return "nvidia-smi n'est pas disponible. Vérifiez l'installation des pilotes NVIDIA."

def verify_cuda():
    print("=== Configuration Python ===")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    print("\n=== Configuration PyTorch ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch est compilé avec CUDA: {torch.version.cuda}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"Nombre de GPUs: {torch.cuda.device_count()}")
        print(f"Appareil actuel: {torch.cuda.current_device()}")
        print(f"Nom du GPU: {torch.cuda.get_device_name(0)}")
    
    print("\n=== Test simple GPU ===")
    if torch.cuda.is_available():
        # Créer et placer un tenseur sur GPU
        x = torch.rand(1000, 1000, device='cuda')
        y = torch.rand(1000, 1000, device='cuda')
        
        # Mesurer le temps d'une opération
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        z = x @ y  # Multiplication matricielle
        end_event.record()
        
        # Synchroniser CUDA
        torch.cuda.synchronize()
        
        print(f"Temps pour multiplication matricielle 1000x1000: {start_event.elapsed_time(end_event):.2f} ms")
        print(f"Résultat sur device: {z.device}")
    else:
        print("Impossible d'effectuer le test GPU - CUDA n'est pas disponible")
    
    print("\n=== nvidia-smi output ===")
    print(get_nvidia_smi_output())

if __name__ == "__main__":
    verify_cuda()
    
    if not torch.cuda.is_available():
        print("\n\nProblèmes possibles:")
        print("1. PyTorch installé sans support CUDA")
        print("   → Solution: Réinstallez PyTorch avec CUDA: https://pytorch.org/get-started/locally/")
        print("2. Pilotes NVIDIA manquants ou obsolètes")
        print("   → Solution: Installez/mettez à jour les pilotes NVIDIA")
        print("3. Version CUDA incompatible avec PyTorch")
        print("   → Solution: Assurez-vous que votre version CUDA est compatible avec votre version PyTorch")