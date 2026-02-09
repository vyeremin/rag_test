import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple

from prompt import build_prompt


MODEL_NAME = "Qwen/Qwen3-0.6B"


def generate_response(prompt: Dict[str, str], use_thinking: bool = False) -> str:
    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype="auto", device_map="auto"
    ).bfloat16()
    model.config.tie_word_embeddings = False

    text = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True, enable_thinking=use_thinking
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2048,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(
        output_ids[:index], skip_special_tokens=True
    ).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    logger = logging.getLogger("inference")
    logger.debug("Thinking content: %s", thinking_content)
    logger.debug("Content: %s", content)
    return content


def generate_response_with_llm(
    query: str,
    results: List[Tuple[int, float]],
    document_mapping: Dict[int, str],
    use_thinking: bool = False,
) -> str:
    prompt = build_prompt(query, results, document_mapping)
    return generate_response(prompt, use_thinking=use_thinking)
