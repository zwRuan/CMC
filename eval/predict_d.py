import os
import sys

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_PATH)

import argparse
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from delta.eval.chat_model import chat_model, DExpertsLlama, chat_model_d
import json
from delta.data_process.data_utils import get_template_and_fix_tokenizer
from tqdm import tqdm
from peft import PeftModel
# from accelerate import dispatch_model

def add_pad_token(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def load_model(model_path):
    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": None,
        "revision": "main",
        "use_auth_token": None,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        **config_kwargs
    )
    # tokenizer = add_pad_token(tokenizer)
    config = AutoConfig.from_pretrained(model_path, **config_kwargs)
    print("loading {}...".format(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        **config_kwargs
    )
    return model,tokenizer

    # return tokenizer

def get_model(args):
    base_model, tokenizer_base = load_model(args.base_model_path)
    # tokenizer = load_model(args.base_model_path)
    template = get_template_and_fix_tokenizer(args.template, tokenizer_base)
    if args.delta_model_path:
        delta_model, tokenizer_delta = load_model(args.delta_model_path)
        _ = get_template_and_fix_tokenizer(args.template, tokenizer_delta)
        
        if 'basenormalization' in args.delta_model_path:
            model = chat_model_d(template,tokenizer_base,base_model,tokenizer_delta,delta_model,args.base2delta_map_path,'basenormalization')
        elif 'normalization' in args.delta_model_path:
            model = chat_model_d(template,tokenizer_base,base_model,tokenizer_delta,delta_model,args.base2delta_map_path,'normalization')
        else:
            model = chat_model_d(template,tokenizer_base,base_model,tokenizer_delta,delta_model,args.base2delta_map_path,'add')
        return model
    else:
        if args.lora_path:
            base_model = PeftModel.from_pretrained(
                base_model,
                args.lora_path,
                torch_dtype=torch.float16,
            )
        model = chat_model(template, tokenizer_base, base_model)
        return model
    # model = chat_model(template, tokenizer)
    # return model


def prepare_dataset(dataset_file_path):
    with open(dataset_file_path, "r") as f:
        predict_data=json.load(f)
    return predict_data

def inference(model, predict_data, stop_sequences):
    res = []
    model_results = []
    predict_out_dir = os.path.join("output", "pred")
    if not os.path.exists(predict_out_dir):
        os.mkdir(predict_out_dir)
    predict_output_dir_name = os.path.join(
        predict_out_dir, args.predicted_out_filename
    )
    for item in tqdm(predict_data, desc="Inference Progress", unit="item"):
        response = model.chat(query=item["instruction"], stop_id_sequences=stop_sequences, do_sample=False, delta_topk=200)
        model_results.append({"instruction":item["instruction"], "output":response.strip()})
        with open(predict_output_dir_name, "w") as fout:
            json.dump(model_results, fout, indent=2)
        res.append(response)
    return res

def dispatch_model(model):
    r"""
    Dispatches a pre-trained model to GPUs with balanced memory.
    Borrowed from: https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/modeling_utils.py#L2803
    """
    if getattr(model, "is_loaded_in_8bit", False) or getattr(
        model, "is_loaded_in_4bit", False
    ):  # do nothing
        return model

    if torch.cuda.device_count() > 1:
        from accelerate import dispatch_model
        from accelerate.utils import infer_auto_device_map, get_balanced_memory

        if model._no_split_modules is None:
            raise ValueError(
                "The model class needs to implement the `_no_split_modules` attribute."
            )

        kwargs = {
            "dtype": model.dtype,
            "no_split_module_classes": model._no_split_modules,
        }
        max_memory = get_balanced_memory(model, **kwargs)
        # Make sure tied weights are tied before creating the device map.
        model.tie_weights()
        device_map = infer_auto_device_map(model, max_memory=max_memory, **kwargs)
        return dispatch_model(model, device_map)
    else:
        return model.cuda()

def main(args):
    model = get_model(args)
    predict_data = prepare_dataset(args.dataset_file_path)
    stop_sequences = ["\n\nComment:"]  # degenerate stuff for llama 2
    stop_sequences = [model.tokenizer_base.encode(" " + x, add_special_tokens=False)[1:] for x in stop_sequences]
    # import pdb; pdb.set_trace()
    result = inference(model, predict_data, stop_sequences)

    predict_out_dir = os.path.join("output", "pred")
    predict_output_dir_name = os.path.join(
        predict_out_dir, args.predicted_out_filename
    )
    print(f"predict_output_dir_name \t{predict_output_dir_name}")

    model_results = []
    with open(predict_output_dir_name, "w") as fout:
        for example, output in zip(predict_data, result):
            example["output"] = output.strip()
            example["generator"] = args.generator_name
            model_results.append(example)
        json.dump(model_results, fout, indent=2)

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generator_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--delta_model_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--delta2base_map_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--base2delta_map_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--ref_base_model_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--ref_finetune_model_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dataset_file_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--template",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--predicted_out_filename",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    main(args)