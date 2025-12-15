# 🎌 Japanese Speaking Trainer

AI-powered Japanese pronunciation trainer with real-time speech recognition, visual feedback, and performance tracking.

## ✨ Features

- **🎤 Real-time Speech Recognition**: Uses OpenAI Whisper AI for accurate Japanese speech-to-text
- **📊 Audio Waveform Visualization**: See your voice in real-time with 40-band frequency analysis
- **🎯 Intelligent Scoring System**: 
  - Phonetic similarity matching
  - Character position validation
  - Japanese particle normalization (は/wa, を/wo)
  - Length-based scoring
- **🚫 Hallucination Detection**: Aggressive filtering of AI hallucinations (YouTube phrases, etc.)
- **📚 Comprehensive Vocabulary**: 1000+ words and phrases with:
  - Kanji, Hiragana, Katakana representations
  - Detailed etymological breakdowns
  - Character-by-character explanations
- **🔄 Forced Retry System**: Must repeat words until achieving 70%+ score
- **💾 Performance Tracking**: SQLite database tracks:
  - Session history and accuracy
  - Per-word statistics
  - Difficulty ratings
  - Learning progress over time
- **🔁 No Duplicates**: Words won't repeat within a session
- **🎨 Beautiful Dark Mode GUI**: Modern, responsive interface with real-time feedback

## 🎯 How It Works

1. **Learn**: See the word in multiple Japanese writing systems
2. **Listen**: Hear native pronunciation via Google TTS
3. **Speak**: Record your pronunciation attempt
4. **Analyze**: AI scores your pronunciation accuracy
5. **Improve**: Retry until you achieve 70%+ accuracy

## 📋 Requirements

```
Python 3.8+
openai-whisper
pykakasi
gtts
pygame
pyaudio
sounddevice
numpy
scipy
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/IbhopTwoSchool/JapaneseSpeakingTrainer.git
cd JapaneseSpeakingTrainer
```

2. Install dependencies:
```bash
pip install openai-whisper pykakasi gtts pygame pyaudio sounddevice numpy scipy
```

3. Run the application:
```bash
python speak_gui.py
```

## 📖 Usage

1. Click **"▶ Start Training"** to begin
2. The app will display a Japanese word with:
   - Kanji/Kana representation
   - Hiragana reading
   - Romaji pronunciation
   - English meaning
   - Etymological breakdown
3. Listen to the pronunciation (waveform shows TTS speaking)
4. Speak the word when recording starts (waveform turns red)
5. See your score and whether you passed (70%+ required)
6. If you fail, the same word repeats until you succeed
7. Continue practicing through the entire vocabulary set

## 🎨 Vocabulary Categories

- **Greetings & Basics** (50+)
- **Numbers & Counting** (100+)
- **Time Expressions** (50+)
- **Family Members** (30+)
- **Colors** (20+)
- **Body Parts** (40+)
- **Food & Drink** (100+)
- **Common Verbs** (100+)
- **Adjectives** (80+)
- **Animals & Nature** (50+)
- **Places & Locations** (40+)
- **Common Phrases** (100+)

## 🔍 Key Features Explained

### Scoring System
- **Phonetic similarity**: Compares sound patterns
- **Character overlap**: Validates correct character usage
- **Position matching**: Ensures characters are in correct order
- **Length penalties**: Penalizes significantly different lengths
- **Particle normalization**: Handles は(wa), へ(e), を(wo) pronunciation

### Hallucination Detection
- Blocks common Whisper hallucinations:
  - YouTube video endings ("ご視聴ありがとうございました")
  - Channel subscription prompts
  - Context-inappropriate greetings
- Length validation (rejects suspiciously long outputs)
- Compression ratio analysis
- English text rejection

### Performance Tracking
- SQLite database stores all attempts
- Track accuracy over time
- Identify difficult words
- Session history
- Word-specific statistics

## 📁 Project Structure

```
JapaneseSpeakingTrainer/
├── speak.py                  # Core trainer logic and voice recognition
├── speak_gui.py              # GUI interface with tkinter
├── vocabulary_expanded.py     # Comprehensive vocabulary database
├── user_stats.py             # Performance tracking system
└── user_performance.db       # SQLite database (created on first run)
```

## 🎓 Learning Tips

1. **Speak clearly** and at moderate volume
2. **Get close** to the microphone for better recognition
3. **Don't rush** - pronunciation accuracy matters more than speed
4. **Practice particles** - は(wa), を(wo), へ(e) have special pronunciations
5. **Use the breakdown** - understand why characters combine that way

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add more vocabulary
- Improve scoring algorithms
- Enhance UI/UX
- Fix bugs
- Add new features

## 📝 License

MIT License - feel free to use and modify for your own learning!

## 🙏 Acknowledgments

- **OpenAI Whisper** - Speech recognition AI
- **pykakasi** - Japanese text conversion
- **gTTS** - Text-to-speech synthesis

## 🐛 Known Issues

- Whisper may hallucinate on very unclear audio
- Some words may be difficult to recognize depending on accent
- First run downloads Whisper models (~500MB)

---

**Happy Learning! がんばって！ (Ganbatte!)**
