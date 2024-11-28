###IMPORTS###
import torch
import time
from pathlib import Path
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

#import warnings
#warnings.catch_warnings(action="ignore")

class SpeechToTextGenerator:
    """
    Base class for transcribing speech into text.
    """
    def __init__(self):
        self.this_dir = Path(__file__).parent.resolve()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model_id = "openai/whisper-small"
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_id, 
                                                          torch_dtype=self.torch_dtype, 
                                                          low_cpu_mem_usage=True, 
                                                          use_safetensors=True)
        #load model to gpu/cpu
        self.setup()

    def setup(self):
        # Set a timer to calculate load times
        generate_start_time = time.time() 

        # Start loading the correct model as set by "tts_method_xtts_local"
        print(f"\033[94mWhisperSTT Local Loading\033[0m {self.model_id} \033[94minto\033[93m {self.device}\033[0m")
        self.model.to(self.device)

        # Create an end timer for calculating load times
        generate_end_time = time.time()
        generate_elapsed_time = generate_end_time - generate_start_time
        print(f"\033[94mWhisper model loaded in \033[93m{generate_elapsed_time:.2f} seconds.\033[0m")

        
    def generate_text_from_audio(self, audio_filepath="", **generate_kwargs):
        if audio_filepath == "":
            audio_filepath = f"{self.this_dir}/audio_outputs/record.wav"
        processor = AutoProcessor.from_pretrained(self.model_id)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device
        )

        if len(generate_kwargs) == 0:
            generate_kwargs = {
                "max_new_tokens": 256,
                "return_timestamps": True
            }

        generate_start_time = time.time()
        result = pipe(audio_filepath, batch_size=1, generate_kwargs=generate_kwargs) #add parameter generate_kwargs=generate_kwargs
        generate_end_time = time.time()
        generated_time = generate_end_time - generate_start_time
        print(f"[WHISPER_TTS] Generated result in \033[93m {generated_time:.2f} seconds.\033[0m")
        print(f"\n\033[092m {result['text']} \033[0m \n")
        return result['text']

###DEBUGGING###
"""
stt = STT_GENERATOR()
stt.generate_text_from_audio()
"""
#To ignore warnings: python -W ignore script.py