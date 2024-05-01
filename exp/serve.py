import argparse
import gradio as gr
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from exp.attn_pplx import FORWARD_FNS


def serve(
    model_name: str = "meta-llama/Llama-2-7b-hf",
    topk: int = 32,
) -> None:
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", device_map="auto")

    # change model forward function to use skip attention
    if type(model) in FORWARD_FNS:
        convert_skip_attn = FORWARD_FNS[type(model)][1]
    else:
        raise ValueError(f"Model type {type(model)} not supported for special attention functions")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    convert_skip_attn(model, topk=topk)

    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)

    gr.Interface.from_pipeline(pipe).launch(share=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--topk", type=int, default=32)

    args = parser.parse_args()
    
    serve(
        model_name=args.model_name,
        topk=args.topk,
    )
