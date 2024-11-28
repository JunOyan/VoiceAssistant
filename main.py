###IMPORTS###
from stt_gen import stt_main, audio_capture_vc
from text_gen.hf_text_generator import ResponseGenerator
from tts_gen import tts_main

class VoiceAssistant:
    '''
    The main body of the voice assistant. 
    The base class contains a STT module, a text generator and a TTS module.
    '''
    def __init__(self, input_device:str, context:str, has_stt=True, has_gen=True, has_tts=True):
        self.has_stt = has_stt
        self.has_tts = has_tts
        self.has_gen = has_gen
        self.is_generating = False
        self.emotion = "neutral"
        print(f"Starting Voice Assistant with configs - has_stt:{has_stt}, has_gen:{has_gen}, has_tts:{has_tts}")
        
        if has_stt:
            self.audio_capture = audio_capture_vc.AudioCapture(input_device)
            self.stt_module = stt_main.SpeechToTextGenerator()

        if has_tts:
            self.tts_module = tts_main.TextToSpeechGenerator()

        if self.has_gen:
            self.generator = ResponseGenerator(context=context)

    def listen_and_transcribe(self, audiopath="") -> str:
        """
        Listens to the audio file from path and return a transcription. The audio file is located in stt_generator/audio_outputs/record.wav by default.
        """
        if not audiopath:
            audiopath = self.audio_capture.filepath
        if not self.has_stt:
            print("[Voice_Assistant] Cannot generate transcription without has_stt enabled.")
            return ""
        #self.audio_capture.run()
        self.is_generating=True
        transcription = self.stt_module.generate_text_from_audio(audio_filepath=audiopath)
        self.is_generating=False
        return transcription
    
    def generate_text_response(self, user_name, prompt) -> str:
        """
        Returns a response from the prompt.
        """
        if not self.has_gen:
            print("[Voice_Assistant] Cannot generate text response without has_gen enabled.")
            return ""
        self.is_generating=True
        response = self.generator.generate_response(prompt=prompt, user_name=user_name, save_to_history=False)
        self.is_generating=False
        return response
    
    def generate_audio_response(self, text, audiopath="") -> None:
        """
        Generates an audio file from the text provided in the audiopath. 
        The audio file is located in tts_generator/tts_output/response.wav by default.
        """
        if not self.has_tts:
            print("[Voice_Assistant] Cannot generate audio without has_tts enabled.")
            return
        self.is_generating=True
        self.tts_module.generate_audio(text, output_file_path=audiopath)
        self.is_generating=False
    
    #Mainly for standalone or debugging
    def generate_full_cycle_response(self):
        if self.has_stt:
            self.audio_capture.run()
            user_input = self.listen_and_transcribe()
        else:
            user_input = input("> ")
        response = self.generate_text_response(user_name="Jun", prompt=user_input)
        self.generate_audio_response(response)

"""
### DEBUGGING ###
with open("bot_context.txt", 'r') as context_file:
    context = context_file.readlines()
voice_assistant = VoiceAssistant(input_device="", has_stt=False, context=context)

### RUN DEBUG ###
if __name__ == "__main__":
    while True:
        try:
            voice_assistant.generate_full_cycle_response()
        except KeyboardInterrupt:
            break
"""
