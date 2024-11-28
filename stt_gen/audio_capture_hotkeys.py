from pynput import keyboard
from pathlib import Path
import pyaudio
import wave

class Audio_Listener(keyboard.Listener):
    def __init__(self, recorder):
        super().__init__(on_press = self.on_press, on_release = self.on_release)
        self.recorder = recorder
    
    def on_press(self, key):
        if key is None: #unknown event
            pass
        elif isinstance(key, keyboard.Key): #special key event
            if key == keyboard.Key.home:
                self.recorder.start()
        elif isinstance(key, keyboard.KeyCode): #alphanumeric key event
            if key.char == 'q': #press q to quit
                if self.recorder.recording:
                    self.recorder.stop()
                raise Exception("Chat stopped.")
                
    def on_release(self, key):
        if key is None:
            pass
        elif isinstance(key, keyboard.Key): #special key event
            if key == keyboard.Key.home:
                self.recorder.stop()
        elif isinstance(key, keyboard.KeyCode): #alphanumeric key event
            pass

class recorder:
    def __init__(self, wavfile, chunksize=2048, dataformat=pyaudio.paInt16, channels=2, rate=44100):
        self.this_dir = Path(__file__).parent.resolve()
        self.filename = str(self.this_dir) + wavfile
        self.chunksize = chunksize
        self.dataformat = dataformat
        self.channels = channels
        self.rate = rate
        self.recording = False
        self.pa = pyaudio.PyAudio()

    def start(self):
        # we call start and stop from the keyboard listener, so we use the asynchronous 
        # version of pyaudio streaming. The keyboard listener must regain control to 
        # begin listening again for the key release.
        if not self.recording:
            self.wf = wave.open(self.filename, 'wb')
            self.wf.setnchannels(self.channels)
            self.wf.setsampwidth(self.pa.get_sample_size(self.dataformat))
            self.wf.setframerate(self.rate)
            
            def callback(in_data, frame_count, time_info, status):
                #file write should be able to keep up with audio data stream (about 1378 Kbps)
                self.wf.writeframes(in_data) 
                return (in_data, pyaudio.paContinue)
            
            self.stream = self.pa.open(format = self.dataformat,
                                       channels = self.channels,
                                       rate = self.rate,
                                       input = True,
                                       stream_callback = callback)
            self.stream.start_stream()
            self.recording = True
            print('recording started...')
    
    def stop(self):
        if self.recording:         
            self.stream.stop_stream()
            self.stream.close()
            self.wf.close()
            
            self.recording = False
            print('recording finished...')
            exit()

def run():
    r = recorder("/audio_outputs/record.wav")
    l = Audio_Listener(r)
    print('Hold Home to record, press q to quit')
    l.start() #keyboard listener is a thread so we start it here
    l.join() #wait for the tread to terminate

###DEBUGGING###
#run()
