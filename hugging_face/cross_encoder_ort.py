import logging
import os
from functools import wraps
from typing import Callable, Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, is_torch_npu_available
from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils import PushToHubMixin

from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.readers import InputExample
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.util import fullname, get_device_name, import_from_string

from optimum.onnxruntime import ORTModelForSequenceClassification
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderOrt(CrossEncoder):
    def __init__(
        self,
        model_name: str,
        num_labels: int = None,
        max_length: int = None,
        device: str = None,
        tokenizer_args: Dict = None,
        automodel_args: Dict = None,
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        local_files_only: bool = False,
        default_activation_function=None,
        classifier_dropout: float = None,
    ) -> None:
        
        CrossEncoder.__init__(self, model_name, num_labels, max_length, device,
                              tokenizer_args, automodel_args, trust_remote_code,
                              revision, local_files_only, default_activation_function,
                              classifier_dropout)
        # if tokenizer_args is None:
        #     tokenizer_args = {}
        # if automodel_args is None:
        #     automodel_args = {}
        # self.config = AutoConfig.from_pretrained(
        #     model_name, trust_remote_code=trust_remote_code, revision=revision, local_files_only=local_files_only
        # )
        # classifier_trained = True
        # if self.config.architectures is not None:
        #     classifier_trained = any(
        #         [arch.endswith("ForSequenceClassification") for arch in self.config.architectures]
        #     )

        # if classifier_dropout is not None:
        #     self.config.classifier_dropout = classifier_dropout

        # if num_labels is None and not classifier_trained:
        #     num_labels = 1

        # if num_labels is not None:
        #     self.config.num_labels = num_labels
        # self.model = AutoModelForSequenceClassification.from_pretrained(
        #     model_name,
        #     config=self.config,
        #     revision=revision,
        #     trust_remote_code=trust_remote_code,
        #     local_files_only=local_files_only,
        #     **automodel_args,
        # )
        
        self.model = ORTModelForSequenceClassification.from_pretrained(
            model_id=model_name,
            config=self.config,
            revision=revision,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            use_merged=True,
            # file_name=os.path.join(model_name, "model.onnx"),
        )
        self.model = self.model.to(self._target_device)
        
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     model_name,
        #     revision=revision,
        #     local_files_only=local_files_only,
        #     trust_remote_code=trust_remote_code,
        #     **tokenizer_args,
        # )
        # self.max_length = max_length

        # if device is None:
        #     device = get_device_name()
        #     logger.info("Use pytorch device: {}".format(device))

        # self._target_device = torch.device(device)

        # if default_activation_function is not None:
        #     self.default_activation_function = default_activation_function
        #     try:
        #         self.config.sbert_ce_default_activation_function = fullname(self.default_activation_function)
        #     except Exception as e:
        #         logger.warning(
        #             "Was not able to update config about the default_activation_function: {}".format(str(e))
        #         )
        # elif (
        #     hasattr(self.config, "sbert_ce_default_activation_function")
        #     and self.config.sbert_ce_default_activation_function is not None
        # ):
        #     self.default_activation_function = import_from_string(self.config.sbert_ce_default_activation_function)()
        # else:
        #     self.default_activation_function = nn.Sigmoid() if self.config.num_labels == 1 else nn.Identity()

    def predict_ort(
        self,
        sentences: List[List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = None,
        num_workers: int = 0,
        activation_fct=None,
        apply_softmax=False,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
    ) -> Union[List[float], np.ndarray, torch.Tensor]:
        
        input_was_string = False
        if isinstance(sentences[0], str):
            sentences = [sentences]
            input_was_string = True

        # inp_dataloader = DataLoader(
        #     sentences,
        #     batch_size=batch_size,
        #     collate_fn=self.smart_batching_collate_text_only,
        #     num_workers=num_workers,
        #     shuffle=False,
        # )
        
        model_inputs = self.tokenizer(
            sentences, padding=True, truncation="longest_first", return_tensors="pt", max_length=self.max_length
            ).to(self._target_device)

        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
            )

        # iterator = inp_dataloader
        # if show_progress_bar:
        #     iterator = tqdm(inp_dataloader, desc="Batches")

        if activation_fct is None:
            activation_fct = self.default_activation_function

        pred_scores = []
        # with torch.inference_mode():
        #     for features in iterator:
        #         model_predictions = self.model(**features, return_dict=True)
        #         logits = activation_fct(model_predictions.logits)

        #         if apply_softmax and len(logits[0]) > 1:
        #             logits = torch.nn.functional.softmax(logits, dim=1)
        #         pred_scores.extend(logits)
        '''
        Batch size, 推理时没必要。
        By default, pipelines will not batch inference for reasons explained in detail here. 
        The reason is that batching is not necessarily faster, and can actually be quite slower in some cases.
        '''
        with torch.inference_mode():
            # model_predictions = self.model(**model_inputs, return_dict=True)
            model_predictions = self.model(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                return_dict=True
            )
            logits = activation_fct(model_predictions.logits)

            if apply_softmax and len(logits[0]) > 1:
                logits = torch.nn.functional.softmax(logits, dim=1)
            pred_scores.extend(logits)

        if self.config.num_labels == 1:
            pred_scores = [score[0] for score in pred_scores]

        if convert_to_tensor:
            pred_scores = torch.stack(pred_scores)
        elif convert_to_numpy:
            pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])

        if input_was_string:
            pred_scores = pred_scores[0]

        return pred_scores

    def rank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        return_documents: bool = False,
        batch_size: int = 32,
        show_progress_bar: bool = None,
        num_workers: int = 0,
        activation_fct=None,
        apply_softmax=False,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
    ) -> List[Dict[Literal["corpus_id", "score", "text"], Union[int, float, str]]]:
        
        query_doc_pairs = [[query, doc] for doc in documents]
        scores = self.predict_ort(
            query_doc_pairs,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            num_workers=num_workers,
            activation_fct=activation_fct,
            apply_softmax=apply_softmax,
            convert_to_numpy=convert_to_numpy,
            convert_to_tensor=convert_to_tensor,
        )

        results = []
        for i in range(len(scores)):
            if return_documents:
                results.append({"corpus_id": i, "score": scores[i], "text": documents[i]})
            else:
                results.append({"corpus_id": i, "score": scores[i]})

        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def save(self, path: str, *, safe_serialization: bool = True, **kwargs) -> None:
        """
        Saves the model and tokenizer to path; identical to `save_pretrained`
        """
        if path is None:
            return

        logger.info("Save model to {}".format(path))
        self.model.save_pretrained(path, safe_serialization=safe_serialization, **kwargs)
        self.tokenizer.save_pretrained(path, **kwargs)

    def save_pretrained(self, path: str, *, safe_serialization: bool = True, **kwargs) -> None:
        """
        Saves the model and tokenizer to path; identical to `save`
        """
        return self.save(path, safe_serialization=safe_serialization, **kwargs)
