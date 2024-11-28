import json
import time
from pathlib import Path
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from pygame import mixer

#MAIN CLASS FOR GENERATING SPEECH
class TextToSpeechGenerator:
    """
    Base class for producing audio response from text.
    """
    def __init__(self):
        self.this_dir = Path(__file__).parent.resolve()
        self.params = self.load_config(self.this_dir / "config" / "tts_config.json")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.xtts_model_path = "xtts_model"
        mixer.init(devicename=self.params["device_output"])
        self.params["low_vram"] = "false" if not torch.cuda.is_available() else self.params["low_vram"]
        self.setup()

    def load_config(self, file_path):
        with open(file_path, "r") as configfile_path:
            configfile_data = json.load(configfile_path)
        return configfile_data

    def setup(self):
        # Set a timer to calculate load times
        generate_start_time = time.time() 

        # Start loading the correct model as set by "tts_method_xtts_local"
        print(f"\033[94mCoqui-tts XTTSv2 Local Loading\033[0m {self.xtts_model_path} \033[94minto\033[93m {self.device}\033[0m")
        self.xtts_load_model()

        # Create an end timer for calculating load times
        generate_end_time = time.time()
        generate_elapsed_time = generate_end_time - generate_start_time
        print(f"\033[94mCoqui-tts Model Loaded in \033[93m{generate_elapsed_time:.2f} seconds.\033[0m")

    def xtts_load_model(self):
        global xtts_model
        config = XttsConfig()
        config_path = self.this_dir / self.xtts_model_path / "config.json"
        vocab_path_dir = self.this_dir / self.xtts_model_path / "vocab.json"
        checkpoint_dir = self.this_dir / self.xtts_model_path

        config.load_json(str(config_path))
        xtts_model = Xtts.init_from_config(config)
        xtts_model.load_checkpoint(
            config,
            checkpoint_dir=str(checkpoint_dir),
            vocab_path=str(vocab_path_dir),
            #use_deepspeed=self.params["deepspeed_activate"],
        )
        xtts_model.to(self.device)

    def unload_model(self):
        print(f"[{self.params['branding']}Model] \033[94mUnloading model \033[0m")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None

    # LOW VRAM - MODEL MOVER VRAM(cuda)<>System RAM(cpu) for Low VRAM setting
    def change_device(self):
        # Check if CUDA is available before performing GPU-related operations
        if torch.cuda.is_available():
            if device == "cuda":
                device = "cpu"
                xtts_model.to(self.device)
                torch.cuda.empty_cache()
            else:
                device = "cuda"
                xtts_model.to(self.device)

    # PLAY GENERATED AUDIO
    def play_audio(self, output_file):
        mixer.music.load(output_file)
        mixer.music.play()
        while mixer.music.get_busy():  # wait for sound file to finish playing
            time.sleep(1)
        mixer.music.unload()

    # TTS VOICE GENERATION METHOD
    def generate_audio(self, text, voice=None, language=None, output_file_path=""):
        if voice == None: voice = self.params["voice"]
        if language == None: language = self.params["language"]
        if output_file_path == "": 
            output_file_path = str(self.this_dir) + self.params["output_folder_wav"] + "response.wav" 
        
        generate_start_time = time.time()  # Record the start time of generating TTS

        # XTTSv2 LOCAL Method Default
        gpt_cond_latent, speaker_embedding = xtts_model.get_conditioning_latents(
            audio_path=[f"{self.this_dir}/voices/{voice}"],
            gpt_cond_len=xtts_model.config.gpt_cond_len,
            max_ref_length=xtts_model.config.max_ref_len,
            sound_norm_refs=xtts_model.config.sound_norm_refs,
        )

        out = xtts_model.inference(
            text,
            language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=float(self.params["local_temperature"]),
            speed=float(self.params["local_speed"]),
            length_penalty=float(xtts_model.config.length_penalty),
            repetition_penalty=float(self.params["local_repetition_penalty"]),
            top_k=int(xtts_model.config.top_k),
            top_p=float(xtts_model.config.top_p),
            enable_text_splitting=True,
        )
        torchaudio.save(output_file_path, torch.tensor(out["wav"]).unsqueeze(0), 24000)

        # Print Generation time and settings
        generate_end_time = time.time()  # Record the end time to generate TTS
        generate_elapsed_time = generate_end_time - generate_start_time
        print(f"[{self.params['branding']}TTSGen] \033[93m{generate_elapsed_time:.2f} seconds in device {self.device}.\033[0m")

        self.play_audio(output_file_path)
"""
###FOR DEBUGGING, RUN FILE###
tts = TextToSpeechGenerator()
while True:
    try:
        time.sleep(1)  # Add a small delay to avoid high CPU usage
        text = input("Write a sentence to convert to audio: ")
        tts.generate_audio(text=text)
        #listen and output here
    except KeyboardInterrupt:       
        tts.unload_model()
        break
"""