from typing import Optional
import modal

MODEL_ALIAS = "gemma2"
MODELS_DIR = f"/{MODEL_ALIAS}"
VOLUME_NAME = f"{MODEL_ALIAS}"

MODEL_NAME = "google/shieldgemma-2b"

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

app = modal.App(f"{MODEL_ALIAS}-non-optimized", image=image)


try:
    volume = modal.Volume.lookup(VOLUME_NAME, create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download models first with modal run download_model.py")


GPU_CONFIG = modal.gpu.A10G(count=1)


@app.cls(gpu=GPU_CONFIG, secrets=[modal.Secret.from_name("huggingface-secret")], volumes={MODELS_DIR: volume},)
class Model:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_directory = MODELS_DIR + "/" + MODEL_NAME
        self.model = AutoModelForCausalLM.from_pretrained(
            model_directory,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_directory)

    @modal.method()
    def generate(self, chat, is_user_prompt=True, enforce_policies=None):
        import torch
        from torch.nn.functional import softmax

        tokenizer = self.tokenizer
        model = self.model

        print("Model: Loaded on device")
        print("Model: Chat {chat}")

        INPUT_POLICIES = {
            "NO_DANGEROUS_CONTENT": "\"No Dangerous Content\": The prompt shall not contain or seek generation of content that harming oneself and/or others (e.g., accessing or building firearms and explosive devices, promotion of terrorism, instructions for suicide).",
            "NO_HARASSMENT": "\"No Harassment\": The prompt shall not contain or seek generation of content that is malicious, intimidating, bullying, or abusive content targeting another individual (e.g., physical threats, denial of tragic events, disparaging victims of violence).",
            "NO_HATE_SPEECH": "\"No Hate Speech\": The prompt shall not contain or seek generation of content that targeting identity and/or protected attributes (e.g., racial slurs, promotion of discrimination, calls to violence against protected groups) or dehumanizing/belittling/vilifying on the bases of race, ethnicity, religion, disability, age, nationality, veteran status, sexual orientation, gender, gender identity, caste, or any other protected status. We do allow content that includes a reference to a protected individual or group in a positive manner.",
            "NO_SEXUAL_CONTENT": "\"No Sexually Explicit Information\": The prompt shall not contain or seek generation of content that contains references to sexual acts or other lewd content (e.g., sexually graphic descriptions, content aimed at causing arousal). Medical or scientific terms related to human anatomy or sex education are permitted."
        }

        OUTPUT_POLICIES = {
            "NO_DANGEROUS_CONTENT": "\"No Dangerous Content\": The chatbot shall not generate content that harming oneself and/or others (e.g., accessing or building firearms and explosive devices, promotion of terrorism, instructions for suicide).",
            "NO_HARASSMENT": "\"No Harassment\": The chatbot shall not generate content that is malicious, intimidating, bullying, or abusive content targeting another individual (e.g., physical threats, denial of tragic events, disparaging victims of violence).",
            "NO_HATE_SPEECH": "\"No Hate Speech\": The chatbot shall not generate content that targeting identity and/or protected attributes (e.g., racial slurs, promotion of discrimination, calls to violence against protected groups) or dehumanizing/belittling/vilifying on the bases of race, ethnicity, religion, disability, age, nationality, veteran status, sexual orientation, gender, gender identity, caste, or any other protected status. We do allow content that includes a reference to a protected individual or group in a positive manner.",
            "NO_SEXUAL_CONTENT": "\"No Sexually Explicit Information\": The chatbot shall not generate content that contains references to sexual acts or other lewd content (e.g., sexually graphic descriptions, content aimed at causing arousal). Medical or scientific terms related to human anatomy or sex education are permitted."
        }

        contructed_guideline = ""
        SELECTED_POLICY_SET = INPUT_POLICIES if is_user_prompt else OUTPUT_POLICIES

        if is_user_prompt:
            enforce_policies = enforce_policies or ["NO_DANGEROUS_CONTENT", "NO_HARASSMENT", "NO_HATE_SPEECH", "NO_SEXUAL_CONTENT"]
            for policy in enforce_policies:
                if contructed_guideline == "":
                    contructed_guideline = SELECTED_POLICY_SET[policy]
                else:
                    contructed_guideline = contructed_guideline + "\n* " + SELECTED_POLICY_SET[policy]
                    
                
        inputs = tokenizer.apply_chat_template(chat, guideline=contructed_guideline, return_tensors="pt", return_dict=True).to(model.device)

        chat_template_display = tokenizer.apply_chat_template(chat, tokenize=False, guideline=contructed_guideline)
        print(f"Model: Chat Template: {chat_template_display}")

        with torch.no_grad():
            logits = model(**inputs).logits

        # Extract the logits for the Yes and No tokens
        vocab = tokenizer.get_vocab()
        selected_logits = logits[0, -1, [vocab['Yes'], vocab['No']]]

        # Convert these logits to a probability with softmax
        probabilities = softmax(selected_logits, dim=0)

        score = probabilities[0].item()

        print(f"Model: Score: {score}")

        return score

    @modal.exit()
    def stop_engine(self):
        if GPU_CONFIG.count > 1:
            import ray

            ray.shutdown()


@app.function(
    keep_warm=1,
    allow_concurrent_inputs=10,
    timeout=60 * 10,
    secrets=[modal.Secret.from_dotenv()],
    volumes={MODELS_DIR: volume}
)
@modal.asgi_app(label="fa-hg-sg2b")
def tgi_app():
    import os

    import fastapi
    from fastapi.middleware.cors import CORSMiddleware

    from typing import List
    from pydantic import BaseModel
    import logging

    TOKEN = os.getenv("TOKEN")
    if TOKEN is None:
        raise ValueError("Please set the TOKEN environment variable")
    
    # Create a logger
    logger = logging.getLogger(MODEL_ALIAS)
    logger.setLevel(logging.DEBUG)

    # Create a handler for logging to stdout
    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.DEBUG)

    # Create a formatter for the log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stdout_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(stdout_handler)
    
    volume.reload()  # ensure we have the latest version of the weights

    app = fastapi.FastAPI()

    http_bearer = fastapi.security.HTTPBearer(
        scheme_name="Bearer Token",
        description="See code for authentication details.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    async def is_authenticated(api_key: str = fastapi.Security(http_bearer)):
        if api_key.credentials != TOKEN:
            raise fastapi.HTTPException(
                status_code=fastapi.status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
            )
        return {"username": "authenticated_user"}
    
    @app.exception_handler(Exception)
    def error_handler(request, exc):
        status_code = 500
        detail = "Internal Server Error"
        logger.exception(exc)
        if isinstance(exc, fastapi.HTTPException):
            status_code = exc.status_code
            detail = exc.detail
        return fastapi.responses.JSONResponse(
            status_code=status_code,
            content={
                "status": status_code,
                "response": {
                    "detail": detail,
                }
            },
        )

    router = fastapi.APIRouter(dependencies=[fastapi.Depends(is_authenticated)])

    class ChatMessages(BaseModel):
        role: str
        content: str

    class ChatClassificationRequestBody(BaseModel):
        score_threshold: Optional[float] = None
        policies: Optional[List[str]] = None
        chat: List[ChatMessages]

  
    @router.post("/v1/chat/classification")
    async def chat_classification_response(body: ChatClassificationRequestBody):
        policies = body.policies
        score_threshold = body.score_threshold or 0.5
        chat = body.model_dump().get("chat",[])

        print("Serving request for chat classification...")
        print(f"Chat: {chat}")
        score = Model().generate.remote(chat, enforce_policies=policies)

        is_unsafe = score > score_threshold

        return {
            "status": 200,
            "response": {
                "class": "unsafe" if is_unsafe else "safe",
                "score": score,
                "applied_policies": policies,
                "score_threshold": score_threshold
            }
        }


    app.include_router(router)
    return app
