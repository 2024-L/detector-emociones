from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="models/v2",
    repo_id="2024-L/detector-emociones-v2",
    repo_type="model",
    commit_message="Subiendo modelo v2 de emociones"
)
print("¡Éxito! Su modelo ya está en la nube de Hugging Face.")