#!/bin/bash
#
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import triton_python_backend_utils as pb_utils
import numpy as np
import torch
from torch.utils.dlpack import from_dlpack

import json
import os
import yaml
# 在这里加上utils.py文件包含的各种工具，直接使用
from utils import *


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        self.model_config = model_config = json.loads(args["model_config"])
        self.max_batch_size = max(model_config["max_batch_size"], 1)

        # # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT0")
        # # Convert Triton types to numpy types
        self.out0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

        self.init_vocab(self.model_config["parameters"])

        # 这里增加加载模型的代码
        punc_config=read_yaml("./model_repo_paraformer_large_offline_punc/scoring/1/config.yaml")
        with open("./model_repo_paraformer_large_offline_punc/scoring/1/tokens_punc.json","r", encoding="utf-8") as f:
            punc_token_list = json.load(f)
        self.converter=TokenIDConverter(punc_token_list)
        self.punc_list=punc_config["model_conf"]["punc_list"]
        self.period = 0
        for i in range(len(self.punc_list)):
            if self.punc_list[i] == ",":
                self.punc_list[i] = "，"
            elif self.punc_list[i] == "?":
                self.punc_list[i] = "？"
            elif self.punc_list[i] == "。":
                self.period = i
        # print("添加逗号的词表成功")
        # 这里加载起来模型，可以不用每次使用都加载一次
        pb_utils.load_model("punc")

    def init_vocab(self, parameters):
        blank_id = 0
        for li in parameters.items():
            key, value = li
            value = value["string_value"]
            if key == "blank_id":
                self.blank_id = int(value)
            elif key == "lm_path":
                lm_path = value
            elif key == "vocabulary":
                self.vocab_dict = self.load_vocab(value)
            if key == "ignore_id":
                ignore_id = int(value)

    def load_vocab(self, vocab_file):
        """
        load lang_char.txt
        """
        with open(str(vocab_file), "rb") as f:
            vocab_list = json.load(f, encoding='utf-8')
        return vocab_list

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them. You
        # should avoid storing any of the input Tensors in the class attributes
        # as they will be overridden in subsequent inference requests. You can
        # make a copy of the underlying NumPy array and store it if it is
        # required.

        total_seq, max_token_num = 0, 0
        assert len(self.vocab_dict) == 8404, len(self.vocab_dict)
        logits_list, token_num_list = [], []

        for request in requests:
            # Perform inference on the request and append it to responses list...
            in_0 = pb_utils.get_input_tensor_by_name(request, "logits")
            in_1 = pb_utils.get_input_tensor_by_name(request, "token_num")

            logits, token_num = from_dlpack(in_0.to_dlpack()), from_dlpack(in_1.to_dlpack()).cpu()
            max_token_num = max(max_token_num, token_num)

            assert logits.shape[0] == 1
            logits_list.append(logits)
            token_num_list.append(token_num)
            total_seq += 1

        logits_batch = torch.zeros(
            len(logits_list),
            max_token_num,
            len(self.vocab_dict),
            dtype=torch.float32,
            device=logits.device,
        )
        token_num_batch = torch.zeros(len(logits_list))

        for i, (logits, token_num) in enumerate(zip(logits_list, token_num_list)):
            logits_batch[i][: int(token_num)] = logits[0][: int(token_num)]
            token_num_batch[i] = token_num

        yseq_batch = logits_batch.argmax(axis=-1).tolist()
        token_int_batch = [list(filter(lambda x: x not in (0, 2), yseq)) for yseq in yseq_batch]

        # 在这里可以加上punc模型，直接batch_size=1,feat_length是token_int_batch

        tokens_batch = [[self.vocab_dict[i] for i in token_int] for token_int in token_int_batch]

         # 在这里输出结果才对
        for tokens in tokens_batch:
            # print("tokens is :",tokens)
            all_tokens = "".join(tokens)    # 这个就是整个字符串，没有引号或者中括号的那种，接入模型即可

        # print(all_tokens)
        # 这里增加添加标点的操作
        split_size=20
        split_text=code_mix_split_words(all_tokens)
        split_text_id = self.converter.tokens2ids(split_text)
        mini_sentences = split_to_mini_sentence(split_text, split_size)     # 20个文字或者单词分成一个小句
        mini_sentences_id = split_to_mini_sentence(split_text_id, split_size)
        # print("分小句子成功")
        # 确保小句子的个数和tokenID的个数一致
        assert len(mini_sentences) == len(mini_sentences_id)
        # 初始化缓存，存储上一个句子没有处理完成的部分
        cache_sent = []
        cache_sent_id = []
        # 用于逐步构建当前句子及其标点符号。
        new_mini_sentence = ""
        new_mini_sentence_punc = []
        cache_pop_trigger_limit = 200
        for mini_sentence_i in range(len(mini_sentences)):
            # 遍历每个小句，将其与缓存中的数据拼接起来
            mini_sentence = mini_sentences[mini_sentence_i]
            mini_sentence_id = mini_sentences_id[mini_sentence_i]
            # print("cache_sent is: ",cache_sent)
            mini_sentence = cache_sent + mini_sentence
            mini_sentence_id = np.array(cache_sent_id + mini_sentence_id, dtype="int32")
            # 将 mini_sentence_id 转换为批量形式（增加一个维度），并记录其长度。
            # data 是传递给模型的输入字典，包含两个键："text"：形状为 (1, n) 的 token ID 矩阵。"text_lengths"：形状为 (1,) 的长度数组。
            # 转成inputs,text_lengths两个输入，循环着输入得到最终结果
            # 构建输入数据
            inputs = [
                pb_utils.Tensor("inputs", mini_sentence_id[None, :].astype(np.int32)),
                pb_utils.Tensor("text_lengths", np.array([[len(mini_sentence_id)]], dtype=np.int32))
            ]
            # print("进入循环成功")

            # 向 Triton 发起推理请求
            infer_request = pb_utils.InferenceRequest(
                model_name='punc',  # 替换为标点模型的实际名称
                requested_output_names=['logits'],  # 替换为输出名
                inputs=inputs
            )

            # 同步发送请求
            response = infer_request.exec()

            if response.has_error():
                print(f"Inference error: {response.error().message()}")
            else:
                output_tensor = pb_utils.get_output_tensor_by_name(response, 'logits')
                y = output_tensor.as_numpy()[0]  # 获取结果
                # print("Punctuated Text:", y)  # 现在可以正常拿到结果了，需要对结果进行解码后处理
                # 使用 np.argmax 提取每个位置的最大概率对应的标点符号索引。
                punctuations = np.argmax(y, axis=-1)
                # print("Punctuations size is:", punctuations.size)
                # print("len mini sentence is:",len(mini_sentence))
                assert punctuations.size == len(mini_sentence)
            # print("调用模型成功，输出结果：",punctuations)

            # Search for the last Period/QuestionMark as cache
            if mini_sentence_i < len(mini_sentences) - 1:
                # 从后向前遍历标点符号，查找句子的结束位置（如句号或问号）。
                sentenceEnd = -1
                last_comma_index = -1
                for i in range(len(punctuations) - 2, 1, -1):
                    if (
                        self.punc_list[punctuations[i]] == "。"
                        or self.punc_list[punctuations[i]] == "？"
                    ):
                        sentenceEnd = i
                        break
                    # 如果未找到结束位置且句子过长，则以逗号作为分割点。
                    if last_comma_index < 0 and self.punc_list[punctuations[i]] == "，":
                        last_comma_index = i

                if (
                    sentenceEnd < 0
                    and len(mini_sentence) > cache_pop_trigger_limit
                    and last_comma_index >= 0
                ):
                    # The sentence it too long, cut off at a comma.
                    sentenceEnd = last_comma_index
                    punctuations[sentenceEnd] = self.period
                # 将未处理完的部分（如未结束的标点符号）保存到缓存中，供下一个小句使用。
                cache_sent = mini_sentence[sentenceEnd + 1 :]
                cache_sent_id = mini_sentence_id[sentenceEnd + 1 :].tolist()
                mini_sentence = mini_sentence[0 : sentenceEnd + 1]
                punctuations = punctuations[0 : sentenceEnd + 1]

            new_mini_sentence_punc += [int(x) for x in punctuations]
            words_with_punc = []
            # 遍历当前小句的每个单词，根据预测的标点符号索引添加对应的标点符号。
            for i in range(len(mini_sentence)):
                if i > 0:
                    if (
                        len(mini_sentence[i][0].encode()) == 1
                        and len(mini_sentence[i - 1][0].encode()) == 1
                    ):
                        mini_sentence[i] = " " + mini_sentence[i]
                words_with_punc.append(mini_sentence[i])
                # _ 表示无标点符号。
                if self.punc_list[punctuations[i]] != "_":
                    words_with_punc.append(self.punc_list[punctuations[i]])
            new_mini_sentence += "".join(words_with_punc)
            # Add Period for the end of the sentence
            new_mini_sentence_out = new_mini_sentence
            new_mini_sentence_punc_out = new_mini_sentence_punc
            # 处理最后一个句子，如果是最后一个句子，确保其以句号或问号结尾。
            if mini_sentence_i == len(mini_sentences) - 1:
                if new_mini_sentence[-1] == "，" or new_mini_sentence[-1] == "、":
                    new_mini_sentence_out = new_mini_sentence[:-1] + "。"
                    new_mini_sentence_punc_out = new_mini_sentence_punc[:-1] + [self.period]
                elif new_mini_sentence[-1] != "。" and new_mini_sentence[-1] != "？":
                    new_mini_sentence_out = new_mini_sentence + "。"
                    new_mini_sentence_punc_out = new_mini_sentence_punc[:-1] + [self.period]
        
        # print("new mini sentence is:",new_mini_sentence_out)
        # print("new mini sentence punc is:",new_mini_sentence_punc_out)

            
        # hyps = [# 这里是编码了所以打印不出来
        #     "".join([t if t != "<space>" else " " for t in tokens]).encode("utf-8")
        #     for tokens in tokens_batch
        # ]
        hyps =[new_mini_sentence_out.encode("utf-8")]
        responses = []
        for i in range(total_seq):
            sents = np.array(hyps[i: i + 1])
            out0 = pb_utils.Tensor("OUTPUT0", sents.astype(self.out0_dtype))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out0])
            responses.append(inference_response)

        # 测试是否可以直接引入工具
        # print_hello()

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
