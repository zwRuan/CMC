import os
import sys

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_PATH)

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    PreTrainedModel
)
import torch
from delta.llm_base.config_parser import get_train_args
from delta.data_process.data_utils import (
    get_dataset,
    preprocess_dataset,
    split_dataset,
)
from delta.llm_base.model_trainer import (
    Seq2SeqPeftTrainer,
    plot_loss,
)

def count_parameters(model: torch.nn.Module):
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes, multiply the number of parameters by 2
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def get_model(model_args):
    anchor_path = model_args.tgt_model_name_or_path
    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": None,
        "revision": "main",
        "use_auth_token": None,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        anchor_path,
        use_fast=False,
        padding_side=model_args.padding_side,
        **config_kwargs
    )
    config = AutoConfig.from_pretrained(anchor_path, **config_kwargs)
    print("loading {}...".format(anchor_path))
    model = AutoModelForCausalLM.from_pretrained(
        anchor_path,
        config=config,
        torch_dtype=torch.bfloat16,
        **config_kwargs
    )

    for name, param in model.named_parameters():
        param.requires_grad = True
        # if 'delta_model' in name:
        #     param.requires_grad = True
        #     # print(f'learnable param: {name}')
        # else:
        #     param.requires_grad = False

    trainable_params, all_param = count_parameters(model)
    print(
        "trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    )
    
    return model,tokenizer

def dispatch_model(model: "PreTrainedModel") -> "PreTrainedModel":
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

def main():
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args()
    # import pdb; pdb.set_trace()
    model, tokenizer = get_model(model_args)
    model = dispatch_model(model)
    dataset = get_dataset(model_args, data_args)
    dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=-100
        if data_args.ignore_pad_token_for_loss
        else tokenizer.pad_token_id,
    )
    print("loading to GPUs...")
    model.to("cuda")

    training_args_dict = training_args.to_dict()
    training_args_dict.update(
        dict(
            generation_max_length=training_args.generation_max_length
            or data_args.max_target_length,
            generation_num_beams=data_args.eval_num_beams
            or training_args.generation_num_beams,
        )
    )
    training_args = Seq2SeqTrainingArguments(**training_args_dict)

    # Initialize our Trainer
    trainer = Seq2SeqPeftTrainer(
        finetuning_args=finetuning_args,
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        **split_dataset(dataset, data_args, training_args)
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
        )
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        trainer.save_model()
        if trainer.is_world_process_zero() and model_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])


if __name__ == '__main__':
    main()