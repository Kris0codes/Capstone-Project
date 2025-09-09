import sounddevice as sd
import soundfile as sf

# Record 5 seconds of audio
duration = 5  # seconds
samplerate = 16000  # Vosk works best with 16kHz
channels = 1  # mono

print("🎤 Recording... Speak now!")
audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='int16')
sd.wait()  # Wait until recording is finished

# Save to WAV file
sf.write("sample.wav", audio, samplerate)
print("✅ Saved as sample.wav")
