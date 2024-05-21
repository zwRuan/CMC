import torch
from transformers import StoppingCriteriaList, StoppingCriteria
from typing import Optional, Dict, Any
import torch.nn.functional as F
from transformers.generation.utils import (
    ModelOutput,
    top_k_top_p_filtering,
    StoppingCriteriaList,
    LogitsProcessorList
)
from delta.data_process.data_utils import get_template_and_fix_tokenizer
import json

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
        return model.to("cuda")

class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            sequence_should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequence_should_be_stopped = True
                    break
            sequences_should_be_stopped.append(sequence_should_be_stopped)
        return all(sequences_should_be_stopped)

class chat_model():
    def __init__(self, template, tokenizer, base_model, delta_model=None, operate_type='add'):
        if torch.cuda.device_count() > 1:
            self.base_model = dispatch_model(base_model)
            self.delta_model = delta_model.to("cuda") if delta_model else None
        else:
            self.base_model = base_model.to("cuda")
            self.delta_model = delta_model.to("cuda") if delta_model else None
        self.operate_type = operate_type
        self.tokenizer = tokenizer
        self.template = template
        self.vanilla_template = get_template_and_fix_tokenizer("vanilla", self.tokenizer)
        self.alpha = 1.0

        self.base_model.eval()
        if delta_model:
            self.delta_model.eval()


    def chat(self, query, stop_id_sequences, max_new_tokens=512, temperature=1.0, top_p=1.0, do_sample=False):
        
        # messages = [
        #     {'role': 'user', 'content': query}
        # ]
        # chat_template = open('chat_templates/'+self.template+'.jinja').read()
        # chat_template = chat_template.replace('    ', '').replace('\n', '')
        # self.tokenizer.chat_template = chat_template
        # prompt=self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # # import pdb; pdb.set_trace()
        # inputs = self.tokenizer([prompt], return_tensors="pt", truncation=False, add_special_tokens=False).to("cuda")
        # prompt_length = len(inputs['input_ids'][0])
        
        if self.delta_model:
            prompt, _ = self.template.encode_oneturn(
                tokenizer=self.tokenizer,
                query=query,
                resp=""
            )
            input_ids = torch.tensor([prompt], device=self.base_model.device)
            prompt_delta, _ = self.template.encode_oneturn(
                tokenizer=self.tokenizer,
                query=query,
                resp=""
            )
            input_ids_delta = torch.tensor([prompt_delta], device=self.base_model.device)
            # import pdb; pdb.set_trace()
        else:
            prompt, _ = self.template.encode_oneturn(
                tokenizer=self.tokenizer,
                query=query,
                resp=""
            )
            input_ids = torch.tensor([prompt], device=self.base_model.device)
        prompt_length = len(input_ids[0])

        stopping_criteria = StoppingCriteriaList([KeyWordsCriteria(stop_id_sequences)]) if stop_id_sequences else None

        if self.delta_model:
            # print(self.tokenizer.decode(prompt,skip_special_tokens=False))
            generation_output, mid_logits = self.generate(
                input_ids=input_ids,
                input_ids_delta=input_ids_delta,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                stopping_criteria=stopping_criteria,
            )
            outputs = generation_output.tolist()[0][prompt_length:]
            response = self.tokenizer.decode(outputs, skip_special_tokens=True)
            # print(response)
            # import pdb; pdb.set_trace()
        else:
            # print(self.tokenizer.decode(prompt))
            generation_output = self.base_model.generate(
                input_ids=input_ids,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                stopping_criteria=stopping_criteria,
            )
            outputs = generation_output.tolist()[0][prompt_length:]
            response = self.tokenizer.decode(outputs, skip_special_tokens=True)
            # print(query)
            # print(response)
            # import pdb; pdb.set_trace()
            mid_logits=None
        return response, mid_logits
        
    
    def forward(
        self,
        base_model_inputs,
        delta_model_inputs,
        return_dict=None
    ):
        base_model_outputs = self.base_model(**base_model_inputs, return_dict=return_dict)
        delta_model_outputs = self.delta_model(**delta_model_inputs, return_dict=return_dict)

        return base_model_outputs, delta_model_outputs
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_ids_delta: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = 100,
        do_sample: bool = False,
        top_p: float = 1.0,
        temperature: float = 1.0,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        **kwargs
    ):
        base_model_kwargs = kwargs.copy()
        delta_model_kwargs = kwargs.copy()

        # delta_model_input_ids = input_ids.to(self.delta_model.device)

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        eos_token_id_tensor = torch.tensor([self.tokenizer.eos_token_id]).to(input_ids.device)
        mid_logits=[]

        for step in range(max_new_tokens):
            # prepare model inputs with past_key_values and attention_mask
            base_model_inputs = self.base_model.prepare_inputs_for_generation(input_ids, **base_model_kwargs)
            delta_model_inputs = self.delta_model.prepare_inputs_for_generation(input_ids_delta, **delta_model_kwargs)

            # DExperts
            base_model_outputs, delta_model_outputs = self.forward(
                base_model_inputs, delta_model_inputs, return_dict=True
            )

            base_model_next_token_logits = base_model_outputs.logits[..., -1, :]
            delta_model_delta_token_logits = delta_model_outputs.logits[..., -1, :]

            # sometimes our experts have extra (irrelevant) tokens at the end of the normal vocabulary
            delta_model_delta_token_logits = delta_model_delta_token_logits[:, :base_model_next_token_logits.shape[-1]]
            base_model_next_token_logits = base_model_next_token_logits[:, :delta_model_delta_token_logits.shape[-1]]

            # import pdb; pdb.set_trace()
            if self.operate_type=='normalization':
                base_model_next_token_logits=F.log_softmax(base_model_next_token_logits,dim=-1)
                delta_model_delta_token_logits=F.log_softmax(delta_model_delta_token_logits,dim=-1)
            elif self.operate_type=='basenormalization':
                base_model_next_token_logits=F.log_softmax(base_model_next_token_logits,dim=-1)


            next_token_logits = (
                base_model_next_token_logits +
                self.alpha * delta_model_delta_token_logits
            )

            mid_logits=None
            # mid_logits.append([base_model_next_token_logits,delta_model_delta_token_logits,next_token_logits])

            # warp logits
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            if top_p < 1.0:
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_p=top_p)

            # import pdb; pdb.set_trace()
            # decode
            # if do_sample:
            #     probs = F.softmax(next_token_logits, dim=-1)
            #     next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            # else:
            #     next_tokens = torch.argmax(next_token_logits, dim=-1)
            # import pdb; pdb.set_trace()

            # probs1 = F.softmax(base_model_next_token_logits, dim=-1)
            # sorted_tensor1, indices1 = torch.sort(probs1[0], descending=True)
            # sorted_tensor2, indices2 = torch.sort(delta_model_delta_token_logits[0], descending=True)
            # probs = F.softmax(next_token_logits, dim=-1)
            # sorted_tensor, indices = torch.sort(probs[0], descending=True)

            next_tokens = torch.argmax(next_token_logits, dim=-1)
            next_tokens = (
                next_tokens * unfinished_sequences +
                self.tokenizer.pad_token_id * (1 - unfinished_sequences)
            )

            # print("base:")
            # for i in range(5):
            #     print(sorted_tensor1[i], base_model_next_token_logits[0][indices1[i]], indices1[i], self.tokenizer.convert_ids_to_tokens([indices1[i]]), torch.where(indices2 == indices1[i]), sorted_tensor2[torch.where(indices2 == indices1[i])])
            # print("="*10)
            # print("delta:")
            # for i in range(5):
            #     print(sorted_tensor2[i], indices2[i], self.tokenizer.convert_ids_to_tokens([indices2[i]]), torch.where(indices1 == indices2[i]), sorted_tensor1[torch.where(indices1 == indices2[i])])
            # print("="*10)
            # print("base+delta")
            # for i in range(5):
            #     t1=torch.where(indices1 == indices[i])
            #     t2=torch.where(indices2 == indices[i])
            #     print(sorted_tensor[i], indices[i], self.tokenizer.convert_ids_to_tokens([indices[i]]), t1, t2)
            # import pdb; pdb.set_trace()

            # update model inputs for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            input_ids_delta = torch.cat([input_ids_delta, next_tokens[:, None]], dim=-1)

            # update kwargs
            base_model_kwargs = self._update_model_kwargs_for_generation(base_model_outputs, base_model_kwargs)
            delta_model_kwargs = self._update_model_kwargs_for_generation(delta_model_outputs, delta_model_kwargs)

            # stopping criteria
            if stopping_criteria and stopping_criteria(input_ids, None):
                break

            # if eos_token was found in one sentence, set sentence to finished
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )
            del base_model_outputs
            del delta_model_outputs
            torch.cuda.empty_cache()

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                break


        return input_ids, mid_logits
    
    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        # update past_key_values
        kwargs["past_key_values"] = self._extract_past_from_model_output(outputs)

        if getattr(outputs, "state", None) is not None:
            kwargs["state"] = outputs.state

        # update attention mask
        if "attention_mask" in kwargs:
            attention_mask = kwargs["attention_mask"]
            kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        return kwargs

    def _extract_past_from_model_output(self, outputs: ModelOutput):
        past_key_values = None
        if "past_key_values" in outputs:
            past_key_values = outputs.past_key_values
        elif "mems" in outputs:
            past_key_values = outputs.mems
        elif "past_buckets_states" in outputs:
            past_key_values = outputs.past_buckets_states

        return past_key_values

class chat_model_d():
    def __init__(self, template, tokenizer_base, base_model, tokenizer_delta, delta_model, token_map_path, operate_type='add'):
        if torch.cuda.device_count() > 1:
            self.base_model = dispatch_model(base_model)
            self.delta_model = dispatch_model(delta_model) if delta_model else None
        else:
            self.base_model = base_model.to("cuda")
            self.delta_model = delta_model.to("cuda") if delta_model else None
        self.operate_type = operate_type
        self.tokenizer_base = tokenizer_base
        self.tokenizer_delta = tokenizer_delta

        with open(token_map_path+"_id.json","r") as f:
            self.token_map=json.load(f)
        if "one2one" in token_map_path:
            self.one2one = True
            self.token_map=torch.tensor(self.token_map).to("cuda")
            self.map_fixed = torch.where(self.token_map == -1, torch.tensor(0), self.token_map)
            with open(token_map_path+"_match.json","r") as f:
                self.token_match=json.load(f)
            self.token_match=torch.tensor(self.token_match).to("cuda")
        else:
            self.one2one = False

        self.template = template
        self.vanilla_template = get_template_and_fix_tokenizer("vanilla", self.tokenizer_base)
        self.alpha = 1.0

        self.base_model.eval()
        if delta_model:
            self.delta_model.eval()


    def chat(self, query, stop_id_sequences, max_new_tokens=512, temperature=1.0, top_p=1.0, do_sample=False, delta_topk=1000):
        
        prompt_base, _ = self.vanilla_template.encode_oneturn(
            tokenizer=self.tokenizer_base,
            query=query,
            resp=""
        )
        input_ids_base = torch.tensor([prompt_base], device=self.base_model.device)
        prompt_length_base = len(input_ids_base[0])

        prompt_delta, _ = self.template.encode_oneturn(
            tokenizer=self.tokenizer_delta,
            query=query,
            resp=""
        )
        input_ids_delta = torch.tensor([prompt_delta], device=self.base_model.device)
        prompt_length_delta = len(input_ids_delta[0])
        

        stopping_criteria = StoppingCriteriaList([KeyWordsCriteria(stop_id_sequences)]) if stop_id_sequences else None

        # print(self.tokenizer_base.decode(input_ids_base[0],skip_special_tokens=False))
        # print(self.tokenizer_delta.decode(input_ids_delta[0],skip_special_tokens=False))
        generation_output = self.generate(
            input_ids_base=input_ids_base,
            input_ids_delta=input_ids_delta,
            prompt_length_base=prompt_length_base,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            delta_topk=delta_topk
        )
        outputs = generation_output.tolist()[0][prompt_length_base:]
        response = self.tokenizer_base.decode(outputs, skip_special_tokens=True)
        # print(response)
        # import pdb; pdb.set_trace()
        return response
    
    def forward(
        self,
        base_model_inputs,
        delta_model_inputs,
        return_dict=None
    ):
        base_model_outputs = self.base_model(**base_model_inputs, return_dict=return_dict)
        delta_model_outputs = self.delta_model(**delta_model_inputs, return_dict=return_dict)

        return base_model_outputs, delta_model_outputs
    
    @torch.no_grad()
    def generate(
        self,
        input_ids_base: Optional[torch.Tensor] = None,
        input_ids_delta: Optional[torch.Tensor] = None,
        prompt_length_base: Optional[int] = 100,
        max_new_tokens: Optional[int] = 100,
        do_sample: bool = False,
        top_p: float = 1.0,
        temperature: float = 1.0,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        delta_topk: int = 1000,
        **kwargs
    ):
        base_model_kwargs = kwargs.copy()
        delta_model_kwargs = kwargs.copy()
        original_input_ids_delta = input_ids_delta.clone()

        # delta_model_input_ids = input_ids.to(self.delta_model.device)

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids_base.shape[0], dtype=torch.long, device=input_ids_base.device)
        eos_token_id_tensor = torch.tensor([self.tokenizer_base.eos_token_id]).to(input_ids_base.device)

        for step in range(max_new_tokens):
            # prepare model inputs with past_key_values and attention_mask
            base_model_inputs = self.base_model.prepare_inputs_for_generation(input_ids_base, **base_model_kwargs)
            delta_model_inputs = self.delta_model.prepare_inputs_for_generation(input_ids_delta, **delta_model_kwargs)

            base_model_outputs, delta_model_outputs = self.forward(
                base_model_inputs, delta_model_inputs, return_dict=True
            )

            base_model_next_token_logits = base_model_outputs.logits[..., -1, :]
            delta_model_delta_token_logits = delta_model_outputs.logits[..., -1, :]

            if self.operate_type=='normalization':
                base_model_next_token_logits=F.log_softmax(base_model_next_token_logits,dim=-1)
                delta_model_delta_token_logits=F.log_softmax(delta_model_delta_token_logits,dim=-1)
            elif self.operate_type=='basenormalization':
                base_model_next_token_logits=F.log_softmax(base_model_next_token_logits,dim=-1)

            if not self.one2one:
                if step==0:
                    base_topk=100
                else:
                    base_topk=50
                _, base_logits_indices = torch.sort(base_model_next_token_logits[0], descending=True)
                next_token_logits = torch.full_like(base_model_next_token_logits[0], torch.finfo(torch.bfloat16).min)

                for i in range(base_topk):
                    base_token_id = base_logits_indices[i]
                    delta_token_id = self.token_map[base_token_id]
                    if len(delta_token_id):
                        max_delta = max(delta_model_delta_token_logits[0][delta_token_id])
                    else:
                        max_delta = 0
                    next_token_logits[base_token_id]=base_model_next_token_logits[0][base_token_id]+max_delta
                next_token_logits=next_token_logits.unsqueeze(0)  
            else:
                select_delta_model_delta_token_logits = delta_model_delta_token_logits[:, self.map_fixed]
                select_delta_model_delta_token_logits[:, self.token_map == -1] = 0
                next_token_logits = (
                    base_model_next_token_logits +
                    self.alpha * select_delta_model_delta_token_logits
                )

            # warp logits
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            if top_p < 1.0:
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_p=top_p)

            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            # sorted_tensor1, indices1 = torch.sort(base_model_next_token_logits[0], descending=True)
            # sorted_tensor2, indices2 = torch.sort(delta_model_delta_token_logits[0], descending=True)
            # probs = F.softmax(next_token_logits[0], dim=-1)
            # sorted_tensor, indices = torch.sort(next_token_logits[0], descending=True)
            # print("base:")
            # for i in range(5):
            #     print(sorted_tensor1[i], indices1[i], self.tokenizer_base.convert_ids_to_tokens([indices1[i]]))
            # print("="*30)
            # print("delta:")
            # for i in range(5):
            #     print(sorted_tensor2[i], indices2[i], self.tokenizer_delta.convert_ids_to_tokens([indices2[i]]))
            # print("="*30)
            # print("base+delta")
            # for i in range(5):
            #     print(sorted_tensor[i], indices[i], self.tokenizer_base.convert_ids_to_tokens([indices[i]]), torch.where(indices1 == indices[i]),end='')
            #     if self.token_map[indices[i]]!=-1:
            #         print(self.tokenizer_delta.convert_ids_to_tokens([self.token_map[indices[i]]]), delta_model_delta_token_logits[0][self.token_map[indices[i]]],end='')
            #     print('')
            
            next_tokens = torch.argmax(next_token_logits, dim=-1)
            next_tokens = (
                next_tokens * unfinished_sequences +
                self.tokenizer_base.pad_token_id * (1 - unfinished_sequences)
            )

            re_encode_flag=0
            # update model inputs for next step
            input_ids_base = torch.cat([input_ids_base, next_tokens[:, None]], dim=-1)
            delta_tokens=self.token_map[next_tokens]
            if (not self.one2one and len(delta_tokens)==1) or (self.one2one and self.token_match[next_tokens]==1):
                next_tokens_delta=torch.tensor(delta_tokens,device=input_ids_delta.device).unsqueeze(0)
                input_ids_delta = torch.cat([input_ids_delta, next_tokens_delta], dim=-1)
            else:
                # import pdb; pdb.set_trace()
                re_encode_flag=1
                # print("****** re_encode_flag *******")
                outputs_now_base_ids = input_ids_base[0][prompt_length_base:]
                outputs_now_tokens = self.tokenizer_base.decode(outputs_now_base_ids, skip_special_tokens=False)
                outputs_now_delta_ids = torch.tensor(self.tokenizer_delta.encode(outputs_now_tokens)[1:],device=input_ids_delta.device).unsqueeze(0)
                input_ids_delta = torch.cat([original_input_ids_delta, outputs_now_delta_ids],dim=-1)
                # import pdb; pdb.set_trace()
                # base_token=self.tokenizer_base.convert_ids_to_tokens(next_tokens)[0]
                # if base_token=='ifornia':
                #     import pdb; pdb.set_trace()
                # delta_token_ids=self.tokenizer_delta.encode(base_token)[1:]
                # # import pdb; pdb.set_trace()
                # if delta_token_ids[0]==self.tokenizer_delta.convert_tokens_to_ids('▁'):
                #     delta_token_ids=delta_token_ids[1:]
                # next_tokens_delta=torch.tensor(delta_token_ids,device=input_ids_delta.device).unsqueeze(0)
            
            # print("base:")
            # print(self.tokenizer_base.decode(input_ids_base[0],skip_special_tokens=False))
            # print("delta:")
            # print(self.tokenizer_delta.decode(input_ids_delta[0],skip_special_tokens=False))
            # import pdb; pdb.set_trace()
            # delta_model_input_ids = torch.cat([delta_model_input_ids, next_tokens[:, None]], dim=-1)

            # update kwargs
            base_model_kwargs = self._update_model_kwargs_for_generation(base_model_outputs, base_model_kwargs)
            if re_encode_flag==0:
                delta_model_kwargs = self._update_model_kwargs_for_generation(delta_model_outputs, delta_model_kwargs)
            else:
                delta_model_kwargs = {}

            # stopping criteria
            if stopping_criteria and stopping_criteria(input_ids_base, None):
                break

            # if eos_token was found in one sentence, set sentence to finished
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )
            del base_model_outputs
            del delta_model_outputs
            torch.cuda.empty_cache()

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                break


        return input_ids_base
    
    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        # update past_key_values
        kwargs["past_key_values"] = self._extract_past_from_model_output(outputs)

        if getattr(outputs, "state", None) is not None:
            kwargs["state"] = outputs.state

        # update attention mask
        if "attention_mask" in kwargs:
            attention_mask = kwargs["attention_mask"]
            kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        return kwargs

    def _extract_past_from_model_output(self, outputs: ModelOutput):
        past_key_values = None
        if "past_key_values" in outputs:
            past_key_values = outputs.past_key_values
        elif "mems" in outputs:
            past_key_values = outputs.mems
        elif "past_buckets_states" in outputs:
            past_key_values = outputs.past_buckets_states

        return past_key_values


class DExpertsLlama():
    def __init__(self, template, tokenizer, base_model, ref_base_model_path, ref_finetune_model_path):
        if torch.cuda.device_count() > 1:
            self.base = dispatch_model(base_model)
        else:
            self.base = base_model.to("cuda")
        self.antiexpert = ref_base_model_path.to("cuda")
        self.expert = ref_finetune_model_path.to("cuda")

        self.base.eval()
        self.expert.eval()
        self.antiexpert.eval()

        self.tokenizer = tokenizer
        self.template = template
        self.vanilla_template = get_template_and_fix_tokenizer("vanilla", self.tokenizer)
        
        self.alpha = 1.0





    def chat(self, query, stop_id_sequences, max_new_tokens=512, temperature=1.0, top_p=1.0, do_sample=False):
        prompt, _ = self.vanilla_template.encode_oneturn(
            tokenizer=self.tokenizer,
            query=query,
            resp=""
        )
        input_ids = torch.tensor([prompt], device=self.base.device)
        prompt_length = len(input_ids[0])

        prompt_chat, _ = self.template.encode_oneturn(
            tokenizer=self.tokenizer,
            query=query,
            resp=""
        )
        input_ids_chat = torch.tensor([prompt_chat], device=self.base.device)
        # prompt_length = len(input_ids_chat[0])

        stopping_criteria = StoppingCriteriaList([KeyWordsCriteria(stop_id_sequences)]) if stop_id_sequences else None

        # print(self.tokenizer.decode(prompt,skip_special_tokens=False))
        generation_output = self.generate(
            input_ids=input_ids,
            input_ids_chat=input_ids_chat,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
        )
        outputs = generation_output.tolist()[0][prompt_length:]
        response = self.tokenizer.decode(outputs, skip_special_tokens=True)
        # print(response)
        # import pdb; pdb.set_trace()
        return response, None
    
    def forward(
        self,
        base_inputs,
        expert_inputs,
        antiexpert_inputs,
        return_dict=None
    ):
        base_outputs = self.base(**base_inputs, return_dict=return_dict)
        expert_outputs = self.expert(**expert_inputs, return_dict=return_dict)
        antiexpert_outputs = self.antiexpert(**antiexpert_inputs, return_dict=return_dict)

        return base_outputs, expert_outputs, antiexpert_outputs
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_ids_chat: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = 100,
        do_sample: bool = False,
        top_p: float = 1.0,
        temperature: float = 1.0,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        **kwargs
    ):
        base_kwargs = kwargs.copy()
        expert_kwargs = kwargs.copy()
        antiexpert_kwargs = kwargs.copy()

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        eos_token_id_tensor = torch.tensor([self.tokenizer.eos_token_id]).to(input_ids.device)

        for step in range(max_new_tokens):
            # prepare model inputs with past_key_values and attention_mask
            base_inputs = self.base.prepare_inputs_for_generation(input_ids, **base_kwargs)
            expert_inputs = self.expert.prepare_inputs_for_generation(input_ids_chat, **expert_kwargs)
            antiexpert_inputs = self.antiexpert.prepare_inputs_for_generation(input_ids, **antiexpert_kwargs)

            # DExperts
            base_outputs, expert_outputs, antiexpert_outputs = self.forward(
                base_inputs, expert_inputs, antiexpert_inputs, return_dict=True
            )

            base_next_token_logits = base_outputs.logits[..., -1, :]
            expert_next_token_logits = expert_outputs.logits[..., -1, :]
            antiexpert_next_token_logits = antiexpert_outputs.logits[..., -1, :]
            
            # sometimes our experts have extra (irrelevant) tokens at the end of the normal vocabulary
            expert_next_token_logits = expert_next_token_logits[:, :base_next_token_logits.shape[-1]]

            # DExperts!
            next_token_logits = (
                base_next_token_logits +
                self.alpha * (expert_next_token_logits - antiexpert_next_token_logits)
            )

            # warp logits
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            if top_p < 1.0:
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_p=top_p)

            # decode
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            next_tokens = (
                next_tokens * unfinished_sequences +
                self.tokenizer.pad_token_id * (1 - unfinished_sequences)
            )
            # import pdb; pdb.set_trace()
            # update model inputs for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None].to(input_ids.device)], dim=-1)
            input_ids_chat = torch.cat([input_ids_chat, next_tokens[:, None]], dim=-1)

            # update kwargs
            base_kwargs = self._update_model_kwargs_for_generation(base_outputs, base_kwargs)
            expert_kwargs = self._update_model_kwargs_for_generation(expert_outputs, expert_kwargs)
            antiexpert_kwargs = self._update_model_kwargs_for_generation(antiexpert_outputs, antiexpert_kwargs)

            # stopping criteria
            if stopping_criteria and stopping_criteria(input_ids, None):
                break

            # if eos_token was found in one sentence, set sentence to finished
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                break


        return input_ids
    
    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        # update past_key_values
        kwargs["past_key_values"] = self._extract_past_from_model_output(outputs)

        if getattr(outputs, "state", None) is not None:
            kwargs["state"] = outputs.state

        # update attention mask
        if "attention_mask" in kwargs:
            attention_mask = kwargs["attention_mask"]
            kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        return kwargs

    def _extract_past_from_model_output(self, outputs: ModelOutput):
        past_key_values = None
        if "past_key_values" in outputs:
            past_key_values = outputs.past_key_values
        elif "mems" in outputs:
            past_key_values = outputs.mems
        elif "past_buckets_states" in outputs:
            past_key_values = outputs.past_buckets_states

        return past_key_values
