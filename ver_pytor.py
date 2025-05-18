try:
    import torch
    print("✅ PyTorch está instalado.")
    print(f"Versão do PyTorch: {torch.__version__}")
    print(f"CUDA disponível: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Nome da GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("🖥️ Rodando apenas com CPU (sem GPU ou sem suporte CUDA).")
except ImportError:
    print("❌ PyTorch NÃO está instalado. Use o comando:")
    print("    pip install torch torchvision torchaudio")
