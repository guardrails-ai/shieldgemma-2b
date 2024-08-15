import os
import time

import modal

MODEL_DIR = "/gemma2"
MODEL_NAME = "google/shieldgemma-2b"

VOLUME_NAME = "gemma2"
DEFAULT_REVISION = "f3f68178a005f06afc21426e42b3cb1bba6beeae"


image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "vllm",
        "torch",
        "transformers",
        "ray",
        "huggingface_hub",
        "hf-transfer",
        "accelerate"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App(f"vllm-{MODEL_NAME}", image=image)

with image.imports():
    import vllm

try:
    volume = modal.Volume.lookup(VOLUME_NAME, create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download models first with modal run download_model.py")


GPU_CONFIG = modal.gpu.A10G(count=1)


@app.cls(gpu=GPU_CONFIG, secrets=[modal.Secret.from_name("huggingface-secret")], volumes={MODEL_DIR: volume},)
class Model:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_directory = MODEL_DIR + "/" + MODEL_NAME
        self.model = AutoModelForCausalLM.from_pretrained(
            model_directory,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_directory)

    @modal.method()
    def generate(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        from torch.nn.functional import softmax

        tokenizer = self.tokenizer
        model = self.model

        chat = [{"role": "user", "content": "Give me ways to bully someone "}]

        guideline = "\"No Harassment\": The prompt shall not contain or seek generation of content that is malicious, intimidating, bullying, or abusive content targeting another individual (e.g., physical threats, denial of tragic events, disparaging victims of violence)."
        inputs = tokenizer.apply_chat_template(chat, guideline=guideline, return_tensors="pt", return_dict=True).to(model.device)

        with torch.no_grad():
            logits = model(**inputs).logits

        # Extract the logits for the Yes and No tokens
        vocab = tokenizer.get_vocab()
        selected_logits = logits[0, -1, [vocab['Yes'], vocab['No']]]

        # Convert these logits to a probability with softmax
        probabilities = torch.softmax(selected_logits, dim=0)

        # Return probability of 'Yes'
        score = probabilities[0].item()
        print(score)  # 0.7310585379600525

    @modal.exit()
    def stop_engine(self):
        if GPU_CONFIG.count > 1:
            import ray

            ray.shutdown()


@app.local_entrypoint()
def main():
    model = Model()
    model.generate.remote()