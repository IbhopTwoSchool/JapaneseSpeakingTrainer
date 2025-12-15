"""Quick voice test"""
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import json
import sys

print("🎤 Voice Recognition Test")
print("=" * 50)

# Load model
print("Loading model...")
model = Model("model")
print("✅ Model loaded")

# Setup recognizer
rec = KaldiRecognizer(model, 16000)

print("\n🎤 Recording for 5 seconds... SPEAK NOW!")
print("=" * 50)

# Record audio
recording = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype='int16')
sd.wait()

print("✅ Recording complete. Processing...")

# Process audio
rec.AcceptWaveform(recording.tobytes())
result = json.loads(rec.FinalResult())
text = result.get('text', '')

print("=" * 50)
if text:
    print(f"✅ YOU SAID: '{text}'")
else:
    print("❌ No speech detected")
print("=" * 50)
