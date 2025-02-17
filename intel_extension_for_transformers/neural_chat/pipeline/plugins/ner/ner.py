# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from threading import Thread
import re
import time
import torch
import spacy
from transformers import (
    TextIteratorStreamer,
)
from .utils.utils import (
    enforce_stop_tokens,
    get_current_time
)
from .utils.process_text import process_time, process_entities
from intel_extension_for_transformers.neural_chat.prompts import PromptTemplate


class NamedEntityRecognition():
    """
        NER class to inference with fp32 or bf16 llm models
        Set bf16=True if you want to inference with bf16 model.
    """

    def __init__(self, spacy_model="en_core_web_lg") -> None:
        # initialize tokenizer and models
        self.nlp = spacy.load(spacy_model)
        print("[NER info] Spacy model initialized.")


    def ner_inference(self, response):
        start_time = time.time()
        cur_time = get_current_time()
        print("[NER inference] Current time is:{}".format(cur_time))
        text = enforce_stop_tokens(response)
        doc = self.nlp(text)
        mentioned_time = process_time(text, doc)

        new_doc = self.nlp(response)
        result = process_entities(response, new_doc, mentioned_time)
        print("[NER inference] Inference time consumption: ", time.time() - start_time)

        return result

