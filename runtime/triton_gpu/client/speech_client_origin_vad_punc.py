# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from tritonclient.utils import np_to_triton_dtype
import numpy as np
import math
import soundfile as sf


class OfflineSpeechClient(object):
    def __init__(self, triton_client, model_name, protocol_client, args):
        self.triton_client = triton_client
        self.protocol_client = protocol_client
        self.model_name = model_name

    def recognize(self, wav_file, idx=0):
        waveform, sample_rate = sf.read(wav_file)
        samples = np.array([waveform], dtype=np.float32)
        lengths = np.array([[len(waveform)]], dtype=np.int32)
        # better pad waveform to nearest length here
        # target_seconds = math.cel(len(waveform) / sample_rate)
        # target_samples = np.zeros([1, target_seconds  * sample_rate])
        # target_samples[0][0: len(waveform)] = waveform
        # samples = target_samples
        sequence_id = 10086 + idx
        result = ""
        inputs = [
            self.protocol_client.InferInput(
                "WAV", samples.shape, np_to_triton_dtype(samples.dtype)
            ),
            self.protocol_client.InferInput(
                "WAV_LENS", lengths.shape, np_to_triton_dtype(lengths.dtype)
            ),
        ]
        inputs[0].set_data_from_numpy(samples)
        inputs[1].set_data_from_numpy(lengths)
        outputs = [self.protocol_client.InferRequestedOutput("TRANSCRIPTS")]
        response = self.triton_client.infer(
            self.model_name,
            inputs,
            request_id=str(sequence_id),
            outputs=outputs,
        )
        result = response.as_numpy("TRANSCRIPTS")[0].decode("utf-8")
        return [result]


class StreamingSpeechClient(object):
    def __init__(self, triton_client, model_name, protocol_client, args):
        self.triton_client = triton_client
        self.protocol_client = protocol_client
        self.model_name = model_name
        chunk_size = args.chunk_size
        subsampling = args.subsampling
        context = args.context
        frame_shift_ms = args.frame_shift_ms
        frame_length_ms = args.frame_length_ms
        # for the first chunk
        # we need additional frames to generate
        # the exact first chunk length frames
        # since the subsampling will look ahead several frames
        first_chunk_length = (chunk_size - 1) * subsampling + context
        add_frames = math.ceil((frame_length_ms - frame_shift_ms) / frame_shift_ms)
        first_chunk_ms = (first_chunk_length + add_frames) * frame_shift_ms
        other_chunk_ms = chunk_size * subsampling * frame_shift_ms
        self.first_chunk_in_secs = first_chunk_ms / 1000
        self.other_chunk_in_secs = other_chunk_ms / 1000

    def recognize(self, wav_file, idx=0):
        waveform, sample_rate = sf.read(wav_file)
        wav_segs = []
        full_result = []
        former_stride = ""
        i = 0
        while i < len(waveform):
            if i == 0:
                stride = int(self.first_chunk_in_secs * sample_rate)
                wav_segs.append(waveform[i : i + stride])
            else:
                stride = int(self.other_chunk_in_secs * sample_rate)    #10240个采样点，在1秒中
                wav_segs.append(waveform[i : i + stride])
            i += len(wav_segs[-1])

        sequence_id = idx + 10086
        # simulate streaming
        for idx, seg in enumerate(wav_segs):    #这里的seg每个都是切块好的数据，直接传入就行，那我直接修改音频输入成只要到了seg大小就传入就行
            chunk_len = len(seg)
            if idx == 0:
                chunk_samples = int(self.first_chunk_in_secs * sample_rate)
                expect_input = np.zeros((1, chunk_samples), dtype=np.float32)
            else:
                chunk_samples = int(self.other_chunk_in_secs * sample_rate)
                expect_input = np.zeros((1, chunk_samples), dtype=np.float32)

            expect_input[0][0:chunk_len] = seg
            input0_data = expect_input
            input1_data = np.array([[chunk_len]], dtype=np.int32)

            inputs = [
                self.protocol_client.InferInput(
                    "WAV",
                    input0_data.shape,
                    np_to_triton_dtype(input0_data.dtype),
                ),
                self.protocol_client.InferInput(
                    "WAV_LENS",
                    input1_data.shape,
                    np_to_triton_dtype(input1_data.dtype),
                ),
            ]

            inputs[0].set_data_from_numpy(input0_data)
            inputs[1].set_data_from_numpy(input1_data)

            outputs = [self.protocol_client.InferRequestedOutput("TRANSCRIPTS")]
            outputs_vad1 = [self.protocol_client.InferRequestedOutput("start_time")]
            outputs_vad2 = [self.protocol_client.InferRequestedOutput("end_time")]
            response_vad = self.triton_client.infer(
                self.model_name,
                inputs,
                outputs=outputs_vad1 + outputs_vad2,
                sequence_id=sequence_id,
                sequence_start=idx == 0,
                sequence_end=end,
            )
            if response_vad.as_numpy("start_time")[0] == -1 == response_vad.as_numpy("end_time")[0] :
                print("是空块")
                if former_start == former_end == -1:
                    continue
                else:
                    inputs_punc[0]
            end = False
            if idx == len(wav_segs) - 1:
                end = True

            response = self.triton_client.infer(
                self.model_name,
                inputs,
                outputs=outputs,
                sequence_id=sequence_id,
                sequence_start=idx == 0,
                sequence_end=end,
            )
            idx += 1
            result = response.as_numpy("TRANSCRIPTS")[0].decode("utf-8")
            print("Get response from {}th chunk: {}".format(idx, result))
            full_result.append(result)
            former_stride = result
            former_start = response_vad.as_numpy("start_time")[0]
            former_end = response_vad.as_numpy("end_time")[0]
        return [result]
