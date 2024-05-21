from transformers.models.llama.modeling_llama import LlamaForCausalLM, LLAMA_INPUTS_DOCSTRING, _CONFIG_FOR_DOC
from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
from typing import Optional, List, Union, Tuple
from torch.nn import CrossEntropyLoss, NLLLoss
import torch.nn.functional as F

class myLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)

    def add_delta_model(self, model, normalization=None):
        self.delta_model = model
        if normalization=="normalization":
            self.normalization = 1
        elif normalization=="basenormalization":
            self.normalization = 2
        else:
            self.normalization = 0
    
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_ids_delta: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_delta: Optional[torch.Tensor] = None,
        prompt_length_base: Optional[int] = None,
        prompt_length_delta: Optional[int] = None,
        input_length_base: Optional[int] = None,
        input_length_delta: Optional[int] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        
        # import pdb; pdb.set_trace()
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        # get delta_model output
        delta_model_Output=self.delta_model(
            input_ids=input_ids,
            attention_mask=attention_mask_delta,
            use_cache=False,
            return_dict=return_dict
        )
        delta_model_logits=delta_model_Output.logits

        if input_ids_delta:
            # import pdb; pdb.set_trace()
            indices_x = torch.arange(input_ids.shape[-1]).unsqueeze(0).expand(input_ids.size(0), -1)
            indices_y = torch.arange(input_ids_delta.shape[-1]).unsqueeze(0).expand(input_ids_delta.size(0), -1)

            index_x = torch.stack((prompt_length_base, input_length_base), dim=1)
            index_y = torch.stack((prompt_length_delta, input_length_delta), dim=1)

            mask_x = (indices_x >= index_x[:, 0].unsqueeze(1)) & (indices_x < index_x[:, 1].unsqueeze(1))
            mask_y = (indices_y >= index_y[:, 0].unsqueeze(1)) & (indices_y < index_y[:, 1].unsqueeze(1))

            logits[mask_x] += delta_model_logits[mask_y]
        else:
            # import pdb; pdb.set_trace()
            if self.normalization == 0:
                logits = logits + delta_model_logits
            elif self.normalization == 1:
                logprobs_base = F.log_softmax(logits, dim=-1)
                logprobs_delta = F.log_softmax(delta_model_logits, dim=-1)
                logits = logprobs_base+logprobs_delta
            else:
                logprobs_base = F.log_softmax(logits, dim=-1)
                logits = logprobs_base+delta_model_logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)

            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)     
            
            loss = loss_fct(shift_logits, shift_labels)



        # print(loss)
        # import pdb; pdb.set_trace()
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    