import modal
import modal.gpu

vllm_image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "vllm==0.5.4"
).env({
    "VLLM_ATTENTION_BACKEND": "FLASHINFER"
}).pip_install(
    "flashinfer", index_url="https://flashinfer.ai/whl/cu121/torch2.3"
)

MODELS_DIR = "/gemma2"
VOLUME_NAME = "gemma2"

MODEL_NAME = "google/shieldgemma-2b"


N_GPU = 1  

MINUTES = 60 
HOURS = 60 * MINUTES

app = modal.App("shieldgemma-2b")



try:
    volume = modal.Volume.lookup(VOLUME_NAME, create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download models first with modal run download_model.py")


def get_model_config(engine):
    import asyncio

    try:  # adapted from vLLM source -- https://github.com/vllm-project/vllm/blob/507ef787d85dec24490069ffceacbd6b161f4f72/vllm/entrypoints/openai/api_server.py#L235C1-L247C1
        event_loop = asyncio.get_running_loop()
    except RuntimeError:
        event_loop = None

    if event_loop is not None and event_loop.is_running():
        # If the current is instanced by Ray Serve,
        # there is already a running event loop
        model_config = event_loop.run_until_complete(engine.get_model_config())
    else:
        # When using single vLLM without engine_use_ray
        model_config = asyncio.run(engine.get_model_config())

    return model_config

@app.function(
    image=vllm_image,
    gpu=modal.gpu.A10G(count=1),
    container_idle_timeout=5 * MINUTES,
    timeout=24 * HOURS,
    allow_concurrent_inputs=100,
    volumes={MODELS_DIR: volume},
    secrets=[modal.Secret.from_dotenv()]
    
)
@modal.asgi_app()
def serve():
    import fastapi
    import os
    import vllm.entrypoints.openai.api_server as api_server
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.entrypoints.logger import RequestLogger
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from vllm.entrypoints.openai.serving_completion import (
        OpenAIServingCompletion,
    )
    from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
    from vllm.entrypoints.openai.serving_tokenization import OpenAIServingTokenization
    from vllm.usage.usage_lib import UsageContext

    TOKEN = os.getenv("TOKEN")

    volume.reload()  # ensure we have the latest version of the weights

    # create a fastAPI app that uses vLLM's OpenAI-compatible router
    web_app = fastapi.FastAPI(
        title=f"OpenAI-compatible {MODEL_NAME} server",
        description="Run an OpenAI-compatible LLM server with vLLM on modal.com",
        version="0.0.1",
        docs_url="/docs",
    )

    # security: CORS middleware for external requests
    http_bearer = fastapi.security.HTTPBearer(
        scheme_name="Bearer Token",
        description="See code for authentication details.",
    )
    web_app.add_middleware(
        fastapi.middleware.cors.CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


    # security: inject dependency on authed routes
    async def is_authenticated(api_key: str = fastapi.Security(http_bearer)):
        if api_key.credentials != TOKEN:
            raise fastapi.HTTPException(
                status_code=fastapi.status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
            )
        return {"username": "authenticated_user"}

    router = fastapi.APIRouter(dependencies=[fastapi.Depends(is_authenticated)])

    # wrap vllm's router in auth router
    router.include_router(api_server.router)
    # add authed vllm to our fastAPI app
    web_app.include_router(router)

    engine_args = AsyncEngineArgs(
        model=MODELS_DIR + "/" + MODEL_NAME,
        tensor_parallel_size=N_GPU,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        dtype="bfloat16",
        enforce_eager=False,  # capture the graph for faster inference, but slower cold starts (30s > 20s)
    )

    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER
    )

    model_config = get_model_config(engine)

    request_logger = RequestLogger(max_log_len=2048)

    api_server.openai_serving_chat = OpenAIServingChat(
        engine,
        model_config=model_config,
        served_model_names=[MODEL_NAME],
        chat_template=None,
        response_role="assistant",
        lora_modules=[],
        prompt_adapters=[],
        request_logger=request_logger,
    )
    api_server.openai_serving_completion = OpenAIServingCompletion(
        engine,
        model_config=model_config,
        served_model_names=[MODEL_NAME],
        lora_modules=[],
        prompt_adapters=[],
        request_logger=request_logger,
    )
    api_server.openai_serving_embedding = OpenAIServingEmbedding(
        engine,
        model_config,
        served_model_names=[MODEL_NAME],
        request_logger=request_logger,
    )
    api_server.openai_serving_tokenization = OpenAIServingTokenization(
        engine,
        model_config,
        served_model_names=[MODEL_NAME],
        lora_modules=[],
        request_logger=request_logger,
        chat_template=None,
    )

    return web_app