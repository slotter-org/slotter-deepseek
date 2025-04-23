import json
import logging
import os
import threading
from unittest.mock import patch

import pika
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.dynamic_modules_utils import get_imports

logging.basicConfig(
    format="[DeepSeek] %(levelname)s %(asctime)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# -----------ENV---------------------
RABBIT_HOST = os.getenv("RABBIT_HOST", "35.193.7.88")
RABBIT_USER = os.getenv("RABBIT_USER", "rabbitmq-user")
RABBIT_PASS = os.getenv("RABBIT_PASS", "rabbitmq-pass")
REQUEST_QUEUE = os.getenv("REQUEST_QUEUE", "inference_request")
REPLY_EXCHANGE = os.getenv("REPLY_EXCHANGE", "inference_replies")
MODEL_MODE = os.getenv("MODEL_MODE", "production")
PROD_MODEL = os.getenv("PROD_MODEL_NAME", "deepseek-ai/DeepSeek-R1")
DEV_MODEL = os.getenv("DEV_MODEL_NAME", "deepseek-ai/DeepSeek-R1")
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", 2))

# ---------GLOBALS--------------------
sema = threading.Semaphore(MAX_CONCURRENCY)
tokenizer = None
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------MonkeyPatch (Transformers)--
def fixed_get_imports(filename: str) -> list[str]:
    imports = get_imports(filename)
    if filename.endswith("/modeling_deepseek.py"):
        if "flash_attn" in imports:
            imports.remove("flash_attn")
    return imports


# ---------MODEL----------------------
def load_model():
    global tokenizer, model
    name = PROD_MODEL if MODEL_MODE.lower() == "production" else PROD_MODEL
    logger.info("Loading %s model: %s", MODEL_MODE.upper(), name)

    with patch("transformers.dynamic_modules_utils.get_imports", fixed_get_imports):
        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

        if MODEL_MODE.lower() == "production":
            dummy = AutoModelForCausalLM.from_pretrained(
                name, device_map={"": device}, trust_remote_code=True
            )
            dmap = infer_auto_device_map(dummy, no_split_module_classes=["GPTBlock"])
            model = load_checkpoint_and_dispatch(dummy, name, device_map=dmap)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                name, trust_remote_code=True
            ).to(device)

        model.eval()
        logger.info("Model loaded on %s", device)


# -------PROMPT---------------------------
def build_prompt(history):
    return (
        "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in history
        )
        + "\nAssistant:"
    )


# -------STREAM-INTERFACE-----------------
def stream_inference(history, **gen_kw):
    prompt = build_prompt(history)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(
        tokenizer, skip_special_tokens=True, skip_prompt=True
    )
    kwargs = dict(inputs=inputs.input_ids, streamer=streamer, **gen_kw)
    threading.Thread(target=model.generate, kwargs=kwargs).start()
    for chunk in streamer:
        yield chunk


# ------RABBIT---------------------------
def on_request(ch, method, props, body):
    if not sema.acquire(blocking=False):
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
        return
    try:
        payload = json.loads(body)
        rid, convo = payload["request_id"], payload.get("conversation", [])
        logger.info("⬇️  %s tokens in — req_id=%s", len(convo), rid)
        for token in stream_inference(convo, max_new_tokens=100, temperature=0.7):
            ch.basic_publish(
                exchange=REPLY_EXCHANGE,
                routing_key=rid,
                body=json.dumps({"request_id": rid, "token": token, "is_final": False}),
            )
        ch.basic_publish(
            exchange=REPLY_EXCHANGE,
            routing_key=rid,
            body=json.dumps({"request_id": rid, "token": "", "is_final": True}),
        )
        ch.basic_ack(delivery_tag=method.delivery_tag)
        logger.info("✅  finished req_id=%s", rid)
    except Exception as e:
        logger.exception("❌  error: %s", e)
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
    finally:
        sema.release()


# ----------MAIN-------------------------
def main():
    load_model()
    creds = pika.PlainCredentials(RABBIT_USER, RABBIT_PASS)
    params = pika.ConnectionParameters(host=RABBIT_HOST, credentials=creds)
    with pika.BlockingConnection(params) as conn:
        ch = conn.channel()
        ch.queue_declare(queue=REQUEST_QUEUE, durable=True)
        ch.exchange_declare(
            exchange=REPLY_EXCHANGE, exchange_type="direct", durable=True
        )
        ch.basic_qos(prefetch_count=MAX_CONCURRENCY)
        ch.basic_consume(queue=REQUEST_QUEUE, on_message_callback=on_request)
        logger.info("🚀  waiting for requests")
        ch.start_consuming()


if __name__ == "__main__":
    main()
