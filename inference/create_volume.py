import modal
import modal.gpu


MODELS_DIR = "/gemma2"
VOLUME_NAME = "gemma2"

MODEL_NAME = "google/shieldgemma-2b"
DEFAULT_REVISION = "f3f68178a005f06afc21426e42b3cb1bba6beeae"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        [
            "huggingface_hub",  # download models from the Hugging Face Hub
            "hf-transfer",  # download models faster with Rust
            "transformers",
            "jinja2"
        ]
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


MINUTES = 60
HOURS = 60 * MINUTES


app = modal.App(
    image=image, secrets=[modal.Secret.from_name("huggingface-secret")]
)

@app.function(
    volumes={MODELS_DIR: volume}, 
    timeout=4 * HOURS,
)
def download_model(model_name, model_revision, force_download=False):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import json
    import os

    volume.reload()

    model_path = MODELS_DIR + "/" + model_name

    if not os.path.exists(model_path):
        print(f"Model {model_name} does not exist in {model_path}, downloading...")
        snapshot_download(
            model_name,
            local_dir=model_path,
            ignore_patterns=[
                "*.pt",
                "*.bin",
                "*.pth",
                "original/*",
            ],  # Ensure safetensors
            revision=model_revision,
            force_download=force_download,
        )
        # Temp Fix for Gemma2 Models
        config_path = model_path + "/config.json"
        with open(config_path, 'r') as file:
            config = json.load(file)
        config["hidden_act"] = "gelu_pytorch_tanh"
        with open(config_path, 'w') as file:
            json.dump(config, file, indent=4)
    else:
        print(f"Model {model_name} already exists in {model_path}, skipping download...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        chat = [{"role": "user", "content": "GR_USER_CONTENT"}]
        guideline = "GR_GUIDELINES"
        chat_template = tokenizer.apply_chat_template(chat, guideline=guideline, tokenize=False)

        #replace GR_GUIDELINES with something I can inject with .format later
        chat_template = chat_template.replace("GR_GUIDELINES", "{GR_GUIDELINES}")
        chat_template = chat_template.replace("GR_USER_CONTENT", "{GR_USER_CONTENT}")        

        with open(model_path + "/chat_template.txt", "w") as f:
            f.write(chat_template)

    volume.commit()


@app.local_entrypoint()
def main(
    model_name: str = MODEL_NAME,
    model_revision: str = DEFAULT_REVISION,
    force_download: bool = False,
):
    download_model.remote(model_name, model_revision, force_download)
