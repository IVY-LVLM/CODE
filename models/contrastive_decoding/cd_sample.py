import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from .cd_utils import contrastive_decoding


from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import transformers

from transformers.generation.utils import (
    GenerateNonBeamOutput,
    EosTokenCriteria,
    logger,
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput
)

#### For Excel Results ###
import os
import pandas as pd
import datetime
from openpyxl import Workbook
from openpyxl.drawing.image import Image as ExcelImage
import io
from PIL import Image

def _sample(
    self,
    input_ids: torch.LongTensor,
    raw_image_data: Optional[Image.Image] = None,
    processor: Optional[Callable[[torch.LongTensor, torch.Tensor], torch.Tensor]] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    output_logits: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:

    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    if eos_token_id is not None:
        logger.warning_once(
            "`eos_token_id` is deprecated in this function and will be removed in v4.41, use"
            " `stopping_criteria=StoppingCriteriaList([EosTokenCriteria(eos_token_id=eos_token_id)])` instead."
            " Otherwise make sure to set `model.generation_config.eos_token_id`",
            FutureWarning,
        )
        stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))
    else:
        # TODO remove when the method is totally private
        # need to get `eos_token_id` and add stopping criteria, so that generation does not go forever
        eos_token_id = [
            criteria.eos_token_id.tolist() for criteria in stopping_criteria if hasattr(criteria, "eos_token_id")
        ]
        eos_token_id = eos_token_id[0] if eos_token_id else None
        if eos_token_id is None and self.generation_config.eos_token_id is not None:
            eos_token_id = self.generation_config.eos_token_id
            stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))
        
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_logits = output_logits if output_logits is not None else self.generation_config.output_logits
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    if "inputs_embeds" in model_kwargs:
        cur_len = model_kwargs["inputs_embeds"].shape[1]
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)


    # init cd
    results = []
    description = []
    use_cd = False
    alt_text = False
    is_start = True
    input_dict = None
    
    if 'input_dict' in model_kwargs:
        input_dict = model_kwargs['input_dict']

    if input_dict:
        if ("contrastive" in input_dict): 
            use_cd = input_dict['contrastive']
        if 'instead_pixel_values' in input_dict:
            alt_text = True
    else:
        if ("contrastive" in model_kwargs): 
            use_cd = model_kwargs['contrastive']

        if 'instead_pixel_values' in model_kwargs:
            alt_text = True


    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        if (alt_text == True) and (use_cd == False):
            model_inputs['instead_pixel_values'] = model_kwargs['instead_pixel_values']

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]
                
        if use_cd:
            if is_start:
                model_kwargs_cd = copy.deepcopy(model_kwargs)
                is_start = False
            ## cd_comments: forward pass of the model with distorted image input
            model_inputs_cd = self.prepare_inputs_for_generation(input_ids, **model_kwargs_cd)
            if input_dict:
                model_inputs_cd['input_dict'] = input_dict
                model_inputs_cd['instead_pixel_values'] = input_dict['instead_pixel_values']
            else:
                model_inputs_cd['instead_pixel_values'] = model_kwargs['instead_pixel_values']

            assert "instead_pixel_values" in model_inputs_cd, "adasd"
            outputs_cd = self(
                **model_inputs_cd,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            next_token_logits_cd = outputs_cd.logits[:, -1, :]
            
            cd_logits, results, description = contrastive_decoding(processor, model_kwargs, next_token_logits, next_token_logits_cd, results, description)
            
            next_token_logits = cd_logits

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )

        ## cd_comments: update model_kwargs_cd for contrastive decoding
        if use_cd:
            model_kwargs_cd = self._update_model_kwargs_for_generation(
                outputs_cd, model_kwargs_cd, is_encoder_decoder=self.config.is_encoder_decoder
            )
            
        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0

    
    if len(results) != 0:
        sheet_name = datetime.datetime.now().strftime("Results_%Y%m%d_%H%M%S")
        excel_path = "./llava-qa90.xlsx"

        # Determine the mode based on whether the file exists
        if not os.path.exists(excel_path):
            with pd.ExcelWriter(excel_path, mode='w', engine='openpyxl') as writer:
                df = pd.DataFrame(results)
                df.to_excel(writer, sheet_name=sheet_name, startrow=2, index=False)

                book = writer.book
                sheet = book[sheet_name]
                
                # Convert the raw image data to bytes
                img_byte_stream = io.BytesIO()
                raw_image_data.save(img_byte_stream, format="PNG")
                img_byte_stream.seek(0)
                
                # Create an ExcelImage object from the image bytes
                excel_image = ExcelImage(img_byte_stream)
                
                # Add the image to the Excel sheet at cell A1
                sheet.add_image(excel_image, 'O1')

        else:
            # If the file exists, use 'append' mode and specify how sheets are handled
            with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='new') as writer:
                df = pd.DataFrame(results)
                df.to_excel(writer, sheet_name=sheet_name, startrow=2, index=False)

                book = writer.book
                sheet = book[sheet_name]
                
                # Convert the raw image data to bytes
                img_byte_stream = io.BytesIO()
                raw_image_data.save(img_byte_stream, format="PNG")
                img_byte_stream.seek(0)
                
                # Create an ExcelImage object from the image bytes
                excel_image = ExcelImage(img_byte_stream)
                
                # Add the image to the Excel sheet at cell A1
                sheet.add_image(excel_image, 'O1')

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids
    

def picd_sampling():
    transformers.generation.utils.GenerationMixin._sample = _sample