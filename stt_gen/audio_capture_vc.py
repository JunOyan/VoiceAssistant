import pyaudio
import wave
import numpy as np
import time
from pathlib import Path

class AudioCapture:
    """
    Base class for capturing audio from an external device.
    """
    def __init__(self, input_device_name="", filepath=""):
        super().__init__()
        if filepath == "":
            this_dir = str(Path(__file__).parent.resolve())
            filepath = this_dir + "/audio_outputs/record.wav"
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        if input_device_name == "":
            device = self.audio.get_default_input_device_info()
            self.device_index = device["index"]
            self.device_name = device["name"]
        else:
            self.device_index = self.find_input_device_index(input_device_name, self.audio)
            self.device_name = input_device_name
        
        self.filepath = filepath
        # Number of samples per frame
        self.chunk = 1024
        # Format of audio stream                                               
        self.format = pyaudio.paInt16                                   
        # Number of audio channels (1 for mono)
        self.channels = 1                                               
        # Sampling rate in Hertz (samples per second)
        self.rate = 24000                                               
        # Threshold for detecting voice
        self.threshold = 330                                            
        # Duration to wait in seconds after voice stops
        self.silence_duration = 3                                       
    
    def is_silent(self, data):
        """Check if the audio data is below the silence threshold."""
        buffer = np.abs(np.frombuffer(data, dtype=np.int16)).mean()
        #Uncomment for debugging
        #print(buffer)
        return buffer < self.threshold

    def find_input_device_index(self, device_name, audio:pyaudio.PyAudio):
        """Find the index of the input device by name."""
        for i in range(audio.get_device_count()):
            dev_info = audio.get_device_info_by_index(i)
            if dev_info["name"] == device_name:
                return i
        raise ModuleNotFoundError("Input device not found.")

    def run(self):
        # Open the stream for audio input
        stream = self.audio.open(format=self.format, channels=self.channels,
                            rate=self.rate, input=True,
                            input_device_index=self.device_index,
                            frames_per_buffer=self.chunk)

        print(f"Listening from {'default device' if self.device_name is None else self.device_name}...")

        recording = False  # Flag to check if we are recording
        frames = []  # List to store audio frames
        silence_start_time = None

        try:
            while True:
                # Read a chunk of data from the microphone
                data = stream.read(self.chunk)

                if not recording and not self.is_silent(data):
                    print("Voice detected. Recording started.")
                    recording = True
                    frames = [data]  # Start recording with this chunk

                elif recording:
                    frames.append(data)
                    if self.is_silent(data):
                        if silence_start_time is None:
                            silence_start_time = time.time()
                        elif time.time() - silence_start_time > self.silence_duration:
                            print("Silence detected. Recording stopped.")
                            break  # Stop recording
                    else:
                        silence_start_time = None  # Reset the silence timer if voice is detected

        finally:
            # Stop and close the stream
            stream.stop_stream()
            stream.close()

            # Save the recorded data to a WAV file if there is any recorded audio
            if frames:
                wf = wave.open(self.filepath, 'wb')
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(frames))
                wf.close()
                print(f"\nAudio saved in {self.filepath}\n")


"""
# Example: Specify a device name (replace with your actual device name if known)
audio_capture = Audio_Capture()
###RUN###
while True:
    try:
        audio_capture.run()
    except KeyboardInterrupt:
        break
"""