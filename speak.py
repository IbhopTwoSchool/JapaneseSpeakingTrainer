"""
Japanese Speaking Trainer - IMPROVED Voice Recognition
Uses Vosk Japanese model for accurate speech recognition
"""
import random
import time
import sys
import json
import os
import zipfile
import urllib.request
from pathlib import Path
from difflib import SequenceMatcher
import tempfile

try:
    import scipy.io.wavfile
    SCIPY_AVAILABLE = True
except:
    SCIPY_AVAILABLE = False

# Initialize colorama for colored output
try:
    from colorama import Fore, Back, Style, init
    init(autoreset=True)  # Auto-reset colors after each print
    COLORAMA_AVAILABLE = True
except:
    # Fallback if colorama not available
    class Fore:
        GREEN = RED = YELLOW = CYAN = MAGENTA = BLUE = WHITE = RESET = ''
    class Back:
        GREEN = RED = YELLOW = CYAN = MAGENTA = BLUE = BLACK = RESET = ''
    class Style:
        BRIGHT = DIM = NORMAL = RESET_ALL = ''
    COLORAMA_AVAILABLE = False

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Import vocabulary
try:
    from vocabulary_expanded import get_full_vocabulary, WORDS, PHRASES
    USE_EXPANDED = True
except ImportError:
    from vocabulary import WORDS, PHRASES
    USE_EXPANDED = False

try:
    from gtts import gTTS
    import pygame
    GTTS_AVAILABLE = True
except:
    GTTS_AVAILABLE = False
    print("[!] gTTS not available")

try:
    from vosk import Model, KaldiRecognizer
    import sounddevice as sd
    import numpy as np
    VOSK_AVAILABLE = True
except:
    VOSK_AVAILABLE = False
    print("[X] Vosk not installed. Run: pip install vosk sounddevice numpy")

try:
    import whisper
    import torch
    WHISPER_AVAILABLE = True
    print("[OK] Whisper (BEST ACCURACY) available!")
except:
    WHISPER_AVAILABLE = False
    print("[!] Whisper not installed. For BEST accuracy: pip install openai-whisper torch")

try:
    from pykakasi import kakasi
    KAKASI_AVAILABLE = True
except:
    KAKASI_AVAILABLE = False
    print("[!] pykakasi not installed. For full kanji support: pip install pykakasi")

class VoiceRecognizer:
    """Multi-engine Japanese voice recognition with Whisper (best) or Vosk (fallback)"""
    def __init__(self):
        self.vosk_model = None
        self.whisper_model = None
        self.use_whisper = False
        self.last_pitch_pattern = None
        self.last_audio = None
        
        # Try to load Whisper first (MOST ACCURATE)
        if WHISPER_AVAILABLE:
            try:
                print("[LOADING] Whisper model (maximum accuracy mode)...")
                print("[INFO] Using 'small' model - better accuracy (may take a moment)")
                # Options: tiny, base, small, medium, large
                # small = significantly better accuracy than base
                # medium = best accuracy without huge resource usage
                self.whisper_model = whisper.load_model("small")  # Upgraded for accuracy
                self.use_whisper = True
                print("[OK] Whisper 'small' model loaded - MAXIMUM ACCURACY MODE!")
            except Exception as e:
                print(f"[!] Whisper load failed: {e}")
                self.use_whisper = False
        
        # Load Vosk as fallback
        if VOSK_AVAILABLE and not self.use_whisper:
            model_path = "model"
            if not os.path.exists(model_path):
                print("\n[DOWNLOAD] Vosk model not found. Downloading...")
                if self.download_model():
                    try:
                        self.vosk_model = Model(model_path)
                        print("[OK] Vosk Japanese model loaded (fallback)")
                    except Exception as e:
                        print(f"[X] Model load failed: {e}")
                        self.vosk_model = None
                else:
                    print("[X] Model download failed")
                    self.vosk_model = None
            else:
                try:
                    self.vosk_model = Model(model_path)
                    print("[OK] Vosk Japanese model loaded (fallback)")
                except Exception as e:
                    print(f"[X] Model load failed: {e}")
                    self.vosk_model = None
    
    def download_model(self):
        """Download Japanese Vosk model"""
        try:
            print("[DOWNLOAD] Downloading vosk-model-small-ja-0.22 (Japanese, ~48MB)")
            print("           This takes 2-3 minutes (one-time only)...")
            
            url = "https://alphacephei.com/vosk/models/vosk-model-small-ja-0.22.zip"
            zip_path = "model.zip"
            
            # Download
            urllib.request.urlretrieve(url, zip_path)
            
            print("[EXTRACT] Extracting model...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            
            os.rename("vosk-model-small-ja-0.22", "model")
            os.remove(zip_path)
            
            print("[OK] Model ready!")
            return True
        except Exception as e:
            print(f"[X] Download failed: {e}")
            return False
    
    def analyze_pitch(self, audio_data, sample_rate):
        """Analyze pitch contour (inflection) from audio"""
        try:
            # Convert to float32 for processing
            audio = audio_data.flatten().astype(np.float32) / 32768.0
            
            # Simple autocorrelation-based pitch detection
            # Focus on energy and contour rather than exact Hz
            frame_length = int(0.025 * sample_rate)  # 25ms frames
            hop_length = int(0.010 * sample_rate)    # 10ms hop
            
            pitches = []
            energies = []
            
            for i in range(0, len(audio) - frame_length, hop_length):
                frame = audio[i:i+frame_length]
                
                # Energy (volume)
                energy = np.sqrt(np.mean(frame**2))
                energies.append(energy)
                
                # Simple zero-crossing rate (proxy for pitch)
                if energy > 0.01:  # Only if there's actual sound
                    zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
                    pitches.append(zcr)
                else:
                    pitches.append(0)
            
            if not pitches or max(energies) < 0.01:
                return None
            
            # Normalize and create inflection pattern
            pitches = np.array(pitches)
            energies = np.array(energies)
            
            # Find voiced regions (where energy > threshold)
            energy_threshold = max(energies) * 0.3
            voiced_mask = energies > energy_threshold
            
            if np.sum(voiced_mask) < 3:
                return None
            
            # Extract pitch contour from voiced regions
            voiced_pitches = pitches[voiced_mask]
            
            # Simplify to rising/falling/flat pattern
            if len(voiced_pitches) < 3:
                return "flat"
            
            # Divide into 3 segments for pattern detection
            third = len(voiced_pitches) // 3
            start_avg = np.mean(voiced_pitches[:third])
            mid_avg = np.mean(voiced_pitches[third:2*third])
            end_avg = np.mean(voiced_pitches[2*third:])
            
            # Determine pattern
            threshold = 0.02  # Sensitivity threshold
            pattern = []
            
            if mid_avg > start_avg * (1 + threshold):
                pattern.append("↑")
            elif mid_avg < start_avg * (1 - threshold):
                pattern.append("↓")
            else:
                pattern.append("→")
            
            if end_avg > mid_avg * (1 + threshold):
                pattern.append("↑")
            elif end_avg < mid_avg * (1 - threshold):
                pattern.append("↓")
            else:
                pattern.append("→")
            
            return "".join(pattern)
            
        except Exception as e:
            print(f"[!] Pitch analysis error: {e}")
            return None
    
    def listen(self, audio_callback=None, expected_word=None):
        """Listen for voice with pitch/inflection analysis - MAXIMUM ACCURACY
        
        Args:
            audio_callback: Optional function to call with audio levels during recording
                          Should accept (levels: list[float]) as parameter
            expected_word: Optional expected word/phrase (romaji) to allow in hallucination detection
        """
        if not self.whisper_model and not self.vosk_model:
            print("\n[X] Voice recognition unavailable")
            return None
            
        try:
            print(f"\n{Back.RED}{Fore.WHITE}{Style.BRIGHT}{'='*50}{Style.RESET_ALL}")
            print(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}>>> RECORDING NOW - SPEAK! <<<{Style.RESET_ALL}")
            print(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}{'='*50}{Style.RESET_ALL}")
            
            # BALANCED: 4 seconds is enough for single words, much faster
            duration = 4  # Reduced from 7 for speed
            sample_rate = 16000
            
            # Setup for real-time audio level capture
            audio_buffer = []
            
            def audio_capture_callback(indata, frames, time_info, status):
                """Called for each audio block during recording"""
                audio_buffer.append(indata.copy())
                
                # Calculate RMS level for waveform visualization
                if audio_callback:
                    # Calculate energy in multiple frequency bands for visual effect
                    audio_chunk = indata.flatten().astype(np.float32) / 32768.0
                    chunk_size = len(audio_chunk) // 40  # 40 bars
                    levels = []
                    
                    for i in range(40):
                        start_idx = i * chunk_size
                        end_idx = min((i + 1) * chunk_size, len(audio_chunk))
                        if start_idx < len(audio_chunk):
                            chunk = audio_chunk[start_idx:end_idx]
                            rms = np.sqrt(np.mean(chunk**2))
                            levels.append(float(rms * 100))  # Scale for visualization
                        else:
                            levels.append(0.0)
                    
                    audio_callback(levels)
            
            # Record audio with real-time callback
            with sd.InputStream(callback=audio_capture_callback,
                              channels=1,
                              samplerate=sample_rate,
                              dtype='int16'):
                sd.sleep(int(duration * 1000))  # Record for duration
            
            # Combine all audio chunks
            audio_data = np.concatenate(audio_buffer, axis=0)
            
            print("[...] Processing with enhanced accuracy...")
            
            # Store audio for pitch analysis
            self.last_audio = audio_data.copy()
            
            # AUDIO ENHANCEMENT: Normalize volume for better recognition
            audio_float = audio_data.flatten().astype(np.float32)
            
            # Remove silence/noise from start and end (but preserve speech)
            energy = np.abs(audio_float)
            energy_threshold = np.max(energy) * 0.05  # Lower threshold to catch quieter speech
            
            # Find speech boundaries with more padding
            speech_indices = np.where(energy > energy_threshold)[0]
            if len(speech_indices) > 0:
                start_idx = max(0, speech_indices[0] - int(0.2 * sample_rate))  # 200ms before
                end_idx = min(len(audio_float), speech_indices[-1] + int(0.5 * sample_rate))  # 500ms after
                audio_float = audio_float[start_idx:end_idx]
            
            # Normalize audio amplitude more conservatively
            max_val = np.max(np.abs(audio_float))
            if max_val > 0:
                audio_float = audio_float * (0.95 / max_val)  # Normalize to 95% to avoid clipping
            
            # Convert back to int16
            audio_processed = (audio_float * 32767).astype(np.int16)
            
            # Analyze pitch pattern (inflection)
            pitch_pattern = self.analyze_pitch(audio_data, sample_rate)
            self.last_pitch_pattern = pitch_pattern
            
            if pitch_pattern:
                print(f"[TONE] Inflection: {pitch_pattern}")
            
            # Use Whisper (MOST ACCURATE) or Vosk (fallback)
            if self.use_whisper and self.whisper_model:
                print("[ENGINE] Using Whisper 'small' model (maximum accuracy)...")
                
                try:
                    # Convert audio to float32 format that Whisper expects
                    audio_float = audio_processed.astype(np.float32) / 32768.0
                    
                    # Pad audio to at least 0.5 seconds for better recognition
                    min_samples = int(0.5 * sample_rate)
                    if len(audio_float) < min_samples:
                        audio_float = np.pad(audio_float, (0, min_samples - len(audio_float)))
                    
                    print("[PROCESSING] Analyzing your speech with AI...")
                    
                    # Transcribe with ANTI-HALLUCINATION settings
                    result = self.whisper_model.transcribe(
                        audio_float,
                        language='ja',  # Force Japanese only
                        task='transcribe',
                        fp16=False,
                        verbose=False,
                        beam_size=3,  # Reduced from 5 - lower beam size = less hallucinations
                        best_of=3,    # Reduced from 5 - fewer candidates = more conservative
                        temperature=0.0,  # ZERO temperature - most conservative, no randomness
                        compression_ratio_threshold=2.0,  # Lower threshold - reject more aggressively
                        logprob_threshold=-0.8,  # Higher threshold - require higher confidence
                        no_speech_threshold=0.7,  # Higher - more aggressive silence detection
                        condition_on_previous_text=False,  # CRITICAL: Don't use context that could introduce hallucinations
                        initial_prompt="単語",  # Simple "word" prompt - complex prompts can trigger hallucinations
                        word_timestamps=True,
                        suppress_tokens=[-1],
                        without_timestamps=False,
                    )
                    
                    recognized_text = result['text'].strip()
                    
                    # REJECT ENGLISH TEXT - This is a Japanese learning app!
                    # Check if text contains primarily English/Latin characters
                    english_chars = sum(1 for c in recognized_text if c.isascii() and c.isalpha())
                    total_chars = len([c for c in recognized_text if c.isalpha()])
                    
                    if total_chars > 0 and english_chars / total_chars > 0.5:
                        print(f"{Fore.RED}[!] English text detected: '{recognized_text}' - Whisper should output Japanese only!{Style.RESET_ALL}")
                        print(f"{Fore.YELLOW}[TIP] Try speaking more clearly or closer to the microphone{Style.RESET_ALL}")
                        return None
                    
                    # AGGRESSIVE HALLUCINATION DETECTION
                    # YouTube video endings (VERY common hallucinations)
                    youtube_hallucinations = [
                        'ご視聴ありがとうございました', 'goshichouarigatougozaimashita',
                        'ご視聴ありがとうございます', 'goshichouarigatougozaimasu',
                        'チャンネル登録', 'channeltourouku', 'channeltorok',
                        'チャンネル登録お願いします', 'チャンネル登録よろしく',
                        'またお会いしましょう', 'mataoaishimashou',
                    ]
                    
                    # Common phrases that are hallucinated WHEN NOT EXPECTED
                    # These are legitimate vocab words, but Whisper hallucinates them on bad audio
                    common_hallucinations = [
                        ('はじめまして', 'hajimemashite'),
                        ('ありがとうございました', 'arigatougozaimashita'),
                        ('ありがとうございます', 'arigatougozaimasu'),
                        ('おはようございます', 'ohayougozaimasu'),
                        ('こんにちは', 'konnichiwa'),
                        ('よろしくお願いします', 'yoroshikuonegaishimasu'),
                        ('お疲れ様でした', 'otsukaresamamdeshita'),
                        ('またね', 'matane'),
                    ]
                    
                    recognized_lower = recognized_text.lower().replace(' ', '').replace('。', '').replace('、', '')
                    
                    # ALWAYS reject YouTube hallucinations (never legitimate in single-word practice)
                    for phrase in youtube_hallucinations:
                        phrase_clean = phrase.lower().replace(' ', '')
                        if phrase_clean in recognized_lower and len(phrase_clean) > 5:
                            print(f"{Fore.RED}[!] HALLUCINATION DETECTED: '{recognized_text}'{Style.RESET_ALL}")
                            print(f"{Fore.YELLOW}    This is a common Whisper hallucination phrase (YouTube/video ending){Style.RESET_ALL}")
                            print(f"{Fore.CYAN}[TIP] Speak louder and more clearly. Hallucinations happen with unclear audio.{Style.RESET_ALL}")
                            return None
                    
                    # For common greetings/phrases, only reject if NOT the expected word
                    if expected_word:
                        expected_clean = expected_word.lower().replace(' ', '')
                        for japanese, romaji in common_hallucinations:
                            japanese_clean = japanese.replace(' ', '')
                            romaji_clean = romaji.lower().replace(' ', '')
                            
                            # If recognized text matches a hallucination phrase
                            if japanese_clean in recognized_lower or romaji_clean in recognized_lower:
                                # But it's NOT the expected word - reject as hallucination
                                if expected_clean not in romaji_clean and expected_clean not in japanese_clean:
                                    print(f"{Fore.RED}[!] HALLUCINATION DETECTED: '{recognized_text}'{Style.RESET_ALL}")
                                    print(f"{Fore.YELLOW}    Expected '{expected_word}' but got common hallucination phrase{Style.RESET_ALL}")
                                    print(f"{Fore.CYAN}[TIP] Speak louder and more clearly.{Style.RESET_ALL}")
                                    return None
                                # Otherwise, it's the expected word - allow it!
                    
                    # Check text length - reject if suspiciously long
                    if len(recognized_text) > 15:  # Single words rarely exceed 15 characters
                        print(f"{Fore.RED}[!] Output too long ({len(recognized_text)} chars): '{recognized_text}'{Style.RESET_ALL}")
                        print(f"{Fore.YELLOW}    Expected a single word, got a phrase - likely hallucination{Style.RESET_ALL}")
                        return None
                    
                    # Check compression ratio from segments
                    if 'segments' in result and len(result['segments']) > 0:
                        for seg in result['segments']:
                            compression = seg.get('compression_ratio', 0)
                            if compression > 2.0:  # Lowered threshold - be more aggressive
                                print(f"{Fore.RED}[!] High compression ratio ({compression:.2f}) - hallucination detected{Style.RESET_ALL}")
                                return None
                    
                    # Check if audio was actually speech (not silence/noise)
                    is_speech = False
                    if 'segments' in result and len(result['segments']) > 0:
                        avg_prob = np.mean([seg.get('avg_logprob', -1) for seg in result['segments']])
                        confidence = np.exp(avg_prob)
                        
                        # Reject very low confidence results as false positives
                        if confidence < 0.3:  # Below 30% confidence is likely noise
                            print(f"{Fore.YELLOW}[!] Low confidence ({confidence:.0%}) - likely no speech or hallucination")
                            return None
                        
                        # Warn on medium-low confidence
                        if confidence < 0.5:
                            print(f"{Fore.YELLOW}[!] Medium-low confidence ({confidence:.0%}) - may be inaccurate")
                        
                        is_speech = True
                    
                    # Show DETAILED recognition info with colors
                    print(f"\n{Fore.CYAN}{Style.BRIGHT}[WHISPER HEARD]{Style.RESET_ALL} {Fore.WHITE}{recognized_text}{Style.RESET_ALL}")
                    
                    if 'segments' in result and len(result['segments']) > 0 and is_speech:
                        avg_prob = np.mean([seg.get('avg_logprob', -1) for seg in result['segments']])
                        confidence = np.exp(avg_prob)
                        
                        # Color code confidence
                        if confidence > 0.7:
                            conf_color = Fore.GREEN
                        elif confidence > 0.5:
                            conf_color = Fore.YELLOW
                        else:
                            conf_color = Fore.RED
                        
                        print(f"{conf_color}[AI CONFIDENCE]{Style.RESET_ALL} {confidence:.1%}")
                        
                        # Show individual word confidences if available
                        for seg in result['segments']:
                            if 'words' in seg:
                                for word_info in seg['words']:
                                    word_text = word_info.get('word', '')
                                    word_prob = word_info.get('probability', 0)
                                    if word_text:
                                        # Color code word confidence
                                        if word_prob > 0.8:
                                            word_color = Fore.GREEN
                                        elif word_prob > 0.6:
                                            word_color = Fore.YELLOW
                                        else:
                                            word_color = Fore.RED
                                        print(f"  {word_color}Word: '{word_text.strip()}' - {word_prob:.0%} confident{Style.RESET_ALL}")
                    
                    if recognized_text:
                        return recognized_text
                    else:
                        print("\n[!] No speech detected - speak LOUDER")
                        return None
                        
                except Exception as e:
                    print(f"[!] Whisper error: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"[!] Falling back to Vosk...")
                    # Fall through to Vosk if Whisper fails
                    pass
            
            elif self.vosk_model:
                print("[ENGINE] Using Vosk (fallback)...")
                
                # Convert processed audio to bytes
                audio_bytes = audio_processed.tobytes()
                
                # Create recognizer with MAXIMUM accuracy settings
                rec = KaldiRecognizer(self.vosk_model, sample_rate)
                rec.SetWords(True)
                rec.SetPartialWords(True)
                rec.SetMaxAlternatives(3)  # Get top 3 alternatives
                
                # Process ENTIRE audio for maximum accuracy
                rec.AcceptWaveform(audio_bytes)
                result = json.loads(rec.FinalResult())
                
                # Extract best recognition result
                recognized_text = None
                confidence = 0
                
                # Try to get the best alternative
                if 'alternatives' in result and len(result['alternatives']) > 0:
                    # Get highest confidence result
                    best = result['alternatives'][0]
                    recognized_text = best.get('text', '')
                    confidence = best.get('confidence', 0)
                    
                    print(f"[CONFIDENCE] {confidence:.0%}")
                    
                    # Show alternatives if low confidence
                    if confidence < 0.7 and len(result['alternatives']) > 1:
                        alts = [alt.get('text', '') for alt in result['alternatives'][1:3]]
                        if any(alts):
                            print(f"[ALTERNATIVES] {' / '.join(filter(None, alts))}")
                elif 'text' in result:
                    recognized_text = result['text']
                
                if recognized_text and recognized_text.strip():
                    return recognized_text.strip()
                else:
                    print("\n[!] No speech detected - speak LOUDER & HOLD MICROPHONE CLOSER")
                    return None
            else:
                print("[X] No recognition engine available")
                return None
                
        except Exception as e:
            print(f"\n[X] Voice error: {e}")
            return None

class JapaneseSpeakingTrainer:
    def __init__(self):
        print("\n[***] Initializing Japanese Speaking Trainer...")
        
        # Initialize voice recognition
        if VOSK_AVAILABLE or WHISPER_AVAILABLE:
            self.voice = VoiceRecognizer()
            if not self.voice.whisper_model and not self.voice.vosk_model:
                print("\n[!] Voice recognition not available")
                self.voice = None
        else:
            self.voice = None
            
        # Initialize Japanese TTS (Google)
        if GTTS_AVAILABLE:
            try:
                pygame.mixer.init()
                self.use_gtts = True
                print("[OK] Japanese TTS ready (Google)")
            except Exception as e:
                print(f"[!] gTTS error: {e}")
                self.use_gtts = False
        else:
            self.use_gtts = False
            
        self.score = 0
        self.attempts = 0
        self.streak = 0
        self.best_streak = 0
        
        # Initialize kakasi for Japanese text conversion
        self.kakasi_converter = None
        if KAKASI_AVAILABLE:
            try:
                kks = kakasi()
                self.kakasi_converter = kks
                print("[OK] Japanese text converter ready")
            except Exception as e:
                print(f"[!] Kakasi init failed: {e}")
        
        # Load vocabulary
        print(f"[LIB] Loading {len(WORDS)} words and {len(PHRASES)} phrases...")
        
        # Load vocabulary
        if USE_EXPANDED:
            all_vocab = get_full_vocabulary()
            # Add hiragana and breakdown to simple format if missing
            real_words = []
            for item in all_vocab:
                word_dict = {
                    'japanese': item.get('kanji') or item.get('hiragana'),
                    'romaji': item['romaji'],
                    'english': item['english']
                }
                if 'hiragana' in item:
                    word_dict['hiragana'] = item['hiragana']
                if 'breakdown' in item:
                    word_dict['breakdown'] = item['breakdown']
                real_words.append(word_dict)
            print(f"[EXPANDED] Using {len(real_words)} words with full representations")
        else:
            # Filter out fake "practice" words (renshuu123, etc.)
            real_words = [w for w in WORDS if not ('practice' in w['english'].lower() and any(c.isdigit() for c in w['english']))]
            print(f"[FILTER] Using {len(real_words)} real words (removed {len(WORDS)-len(real_words)} practice fillers)")
        
        self.content = {
            'words': real_words,
            'phrases': PHRASES if not USE_EXPANDED else []
        }
        
    def speak(self, text):
        """Speak Japanese using native voice"""
        print(f'[SPEAK] {text}')
        
        if self.use_gtts:
            try:
                # Convert romaji to hiragana for accurate TTS pronunciation
                hiragana_text = self.romaji_to_hiragana(text)
                
                tts = gTTS(text=hiragana_text, lang='ja', slow=False)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                    temp_file = fp.name
                    tts.save(temp_file)
                
                pygame.mixer.music.load(temp_file)
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                
                pygame.mixer.music.unload()
                os.unlink(temp_file)
                
            except Exception as e:
                print(f"[!] TTS error: {e}")
    
    def romaji_to_hiragana(self, romaji):
        """Convert romaji to hiragana for accurate TTS pronunciation"""
        # Comprehensive romaji to hiragana conversion
        conversion = {
            # Full words/phrases first
            'konnichiwa': 'こんにちは', 'konnichiha': 'こんにちは',
            'arigatou': 'ありがとう', 'arigato': 'ありがとう',
            'sayounara': 'さようなら', 'sayonara': 'さようなら',
            'ohayou': 'おはよう', 'ohayo': 'おはよう',
            'oyasumi': 'おやすみ',
            'sumimasen': 'すみません',
            'gomennasai': 'ごめんなさい',
            'itadakimasu': 'いただきます',
            'gochisousama': 'ごちそうさま',
            
            # Multi-character combinations (must come before single characters)
            'kya': 'きゃ', 'kyu': 'きゅ', 'kyo': 'きょ',
            'sha': 'しゃ', 'shu': 'しゅ', 'sho': 'しょ', 'shi': 'し',
            'cha': 'ちゃ', 'chu': 'ちゅ', 'cho': 'ちょ', 'chi': 'ち',
            'nya': 'にゃ', 'nyu': 'にゅ', 'nyo': 'にょ',
            'hya': 'ひゃ', 'hyu': 'ひゅ', 'hyo': 'ひょ',
            'mya': 'みゃ', 'myu': 'みゅ', 'myo': 'みょ',
            'rya': 'りゃ', 'ryu': 'りゅ', 'ryo': 'りょ',
            'gya': 'ぎゃ', 'gyu': 'ぎゅ', 'gyo': 'ぎょ',
            'ja': 'じゃ', 'ju': 'じゅ', 'jo': 'じょ', 'ji': 'じ',
            'bya': 'びゃ', 'byu': 'びゅ', 'byo': 'びょ',
            'pya': 'ぴゃ', 'pyu': 'ぴゅ', 'pyo': 'ぴょ',
            'tsu': 'つ', 'tu': 'つ',
            'fu': 'ふ', 'hu': 'ふ',
            
            # Basic hiragana (single syllables)
            'ka': 'か', 'ki': 'き', 'ku': 'く', 'ke': 'け', 'ko': 'こ',
            'sa': 'さ', 'su': 'す', 'se': 'せ', 'so': 'そ',
            'ta': 'た', 'te': 'て', 'to': 'と',
            'na': 'な', 'ni': 'に', 'nu': 'ぬ', 'ne': 'ね', 'no': 'の',
            'ha': 'は', 'hi': 'ひ', 'he': 'へ', 'ho': 'ほ',
            'ma': 'ま', 'mi': 'み', 'mu': 'む', 'me': 'め', 'mo': 'も',
            'ya': 'や', 'yu': 'ゆ', 'yo': 'よ',
            'ra': 'ら', 'ri': 'り', 'ru': 'る', 're': 'れ', 'ro': 'ろ',
            'wa': 'わ', 'wo': 'を', 'nn': 'ん', 'n': 'ん',
            'ga': 'が', 'gi': 'ぎ', 'gu': 'ぐ', 'ge': 'げ', 'go': 'ご',
            'za': 'ざ', 'zu': 'ず', 'ze': 'ぜ', 'zo': 'ぞ',
            'da': 'だ', 'de': 'で', 'do': 'ど',
            'ba': 'ば', 'bi': 'び', 'bu': 'ぶ', 'be': 'べ', 'bo': 'ぼ',
            'pa': 'ぱ', 'pi': 'ぴ', 'pu': 'ぷ', 'pe': 'ぺ', 'po': 'ぽ',
            'a': 'あ', 'i': 'い', 'u': 'う', 'e': 'え', 'o': 'お',
        }
        
        romaji = romaji.lower().strip()
        
        # Check for full phrase match first
        if romaji in conversion:
            return conversion[romaji]
        
        # Convert character by character
        result = []
        i = 0
        while i < len(romaji):
            # Try 3-character combinations
            if i + 3 <= len(romaji) and romaji[i:i+3] in conversion:
                result.append(conversion[romaji[i:i+3]])
                i += 3
            # Try 2-character combinations
            elif i + 2 <= len(romaji) and romaji[i:i+2] in conversion:
                result.append(conversion[romaji[i:i+2]])
                i += 2
            # Try single character
            elif romaji[i] in conversion:
                result.append(conversion[romaji[i]])
                i += 1
            # Skip unknown characters
            else:
                i += 1
        
        hiragana = ''.join(result)
        return hiragana if hiragana else romaji
    
    def japanese_to_romaji(self, text):
        """Convert any Japanese text (kanji/kana) to romaji using kakasi"""
        if not text:
            return ""
        
        # Use kakasi if available for accurate conversion
        if self.kakasi_converter and KAKASI_AVAILABLE:
            try:
                result = self.kakasi_converter.convert(text)
                romaji_parts = [item['hepburn'] for item in result if 'hepburn' in item]
                return ''.join(romaji_parts).lower()
            except Exception as e:
                print(f"[!] Kakasi conversion error: {e}")
                # Fall through to manual conversion
        
        # Fallback: use existing normalize function
        return self.normalize(text)
    
    def normalize(self, text):
        """Normalize text for comparison - with comprehensive Japanese to romaji conversion"""
        if not text:
            return ""
        
        text = text.lower().strip()
        
        # Comprehensive hiragana/katakana -> romaji mapping
        kana_to_romaji = {
            # Complete phrase mappings first
            'こんにちは': 'konnichiha', 'こんにちわ': 'konnichiha',
            'こんばんは': 'konbanha', 'こんばんわ': 'konbanha',
            'ありがとう': 'arigato', 'ありがとお': 'arigato',
            'さようなら': 'sayonara', 'さよなら': 'sayonara',
            'おはよう': 'ohayo', 'おはよお': 'ohayo',
            'おやすみ': 'oyasumi',
            'すみません': 'sumimasen',
            'ごめんなさい': 'gomennasai',
            'いただきます': 'itadakimasu',
            'げんきです': 'genkidesu',
            'おげんきですか': 'ogenkidesuka',
            
            # Hiragana
            'あ': 'a', 'い': 'i', 'う': 'u', 'え': 'e', 'お': 'o',
            'か': 'ka', 'き': 'ki', 'く': 'ku', 'け': 'ke', 'こ': 'ko',
            'さ': 'sa', 'し': 'shi', 'す': 'su', 'せ': 'se', 'そ': 'so',
            'た': 'ta', 'ち': 'chi', 'つ': 'tsu', 'て': 'te', 'と': 'to',
            'な': 'na', 'に': 'ni', 'ぬ': 'nu', 'ね': 'ne', 'の': 'no',
            'は': 'ha', 'ひ': 'hi', 'ふ': 'fu', 'へ': 'he', 'ほ': 'ho',
            'ま': 'ma', 'み': 'mi', 'む': 'mu', 'め': 'me', 'も': 'mo',
            'や': 'ya', 'ゆ': 'yu', 'よ': 'yo',
            'ら': 'ra', 'り': 'ri', 'る': 'ru', 'れ': 're', 'ろ': 'ro',
            'わ': 'wa', 'を': 'wo', 'ん': 'n',
            'が': 'ga', 'ぎ': 'gi', 'ぐ': 'gu', 'げ': 'ge', 'ご': 'go',
            'ざ': 'za', 'じ': 'ji', 'ず': 'zu', 'ぜ': 'ze', 'ぞ': 'zo',
            'だ': 'da', 'ぢ': 'ji', 'づ': 'zu', 'で': 'de', 'ど': 'do',
            'ば': 'ba', 'び': 'bi', 'ぶ': 'bu', 'べ': 'be', 'ぼ': 'bo',
            'ぱ': 'pa', 'ぴ': 'pi', 'ぷ': 'pu', 'ぺ': 'pe', 'ぽ': 'po',
            'きゃ': 'kya', 'きゅ': 'kyu', 'きょ': 'kyo',
            'しゃ': 'sha', 'しゅ': 'shu', 'しょ': 'sho',
            'ちゃ': 'cha', 'ちゅ': 'chu', 'ちょ': 'cho',
            'にゃ': 'nya', 'にゅ': 'nyu', 'にょ': 'nyo',
            'ひゃ': 'hya', 'ひゅ': 'hyu', 'ひょ': 'hyo',
            'みゃ': 'mya', 'みゅ': 'myu', 'みょ': 'myo',
            'りゃ': 'rya', 'りゅ': 'ryu', 'りょ': 'ryo',
            'ぎゃ': 'gya', 'ぎゅ': 'gyu', 'ぎょ': 'gyo',
            'じゃ': 'ja', 'じゅ': 'ju', 'じょ': 'jo',
            'びゃ': 'bya', 'びゅ': 'byu', 'びょ': 'byo',
            'ぴゃ': 'pya', 'ぴゅ': 'pyu', 'ぴょ': 'pyo',
            
            # Katakana (same sounds)
            'ア': 'a', 'イ': 'i', 'ウ': 'u', 'エ': 'e', 'オ': 'o',
            'カ': 'ka', 'キ': 'ki', 'ク': 'ku', 'ケ': 'ke', 'コ': 'ko',
            'サ': 'sa', 'シ': 'shi', 'ス': 'su', 'セ': 'se', 'ソ': 'so',
            'タ': 'ta', 'チ': 'chi', 'ツ': 'tsu', 'テ': 'te', 'ト': 'to',
            'ナ': 'na', 'ニ': 'ni', 'ヌ': 'nu', 'ネ': 'ne', 'ノ': 'no',
            'ハ': 'ha', 'ヒ': 'hi', 'フ': 'fu', 'ヘ': 'he', 'ホ': 'ho',
            'マ': 'ma', 'ミ': 'mi', 'ム': 'mu', 'メ': 'me', 'モ': 'mo',
            'ヤ': 'ya', 'ユ': 'yu', 'ヨ': 'yo',
            'ラ': 'ra', 'リ': 'ri', 'ル': 'ru', 'レ': 're', 'ロ': 'ro',
            'ワ': 'wa', 'ヲ': 'wo', 'ン': 'n',
            'ガ': 'ga', 'ギ': 'gi', 'グ': 'gu', 'ゲ': 'ge', 'ゴ': 'go',
            'ザ': 'za', 'ジ': 'ji', 'ズ': 'zu', 'ゼ': 'ze', 'ゾ': 'zo',
            'ダ': 'da', 'ヂ': 'ji', 'ヅ': 'zu', 'デ': 'de', 'ド': 'do',
            'バ': 'ba', 'ビ': 'bi', 'ブ': 'bu', 'ベ': 'be', 'ボ': 'bo',
            'パ': 'pa', 'ピ': 'pi', 'プ': 'pu', 'ペ': 'pe', 'ポ': 'po',
            'キャ': 'kya', 'キュ': 'kyu', 'キョ': 'kyo',
            'シャ': 'sha', 'シュ': 'shu', 'ショ': 'sho',
            'チャ': 'cha', 'チュ': 'chu', 'チョ': 'cho',
            'ニャ': 'nya', 'ニュ': 'nyu', 'ニョ': 'nyo',
            'ヒャ': 'hya', 'ヒュ': 'hyu', 'ヒョ': 'hyo',
            'ミャ': 'mya', 'ミュ': 'myu', 'ミョ': 'myo',
            'リャ': 'rya', 'リュ': 'ryu', 'リョ': 'ryo',
            'ギャ': 'gya', 'ギュ': 'gyu', 'ギョ': 'gyo',
            'ジャ': 'ja', 'ジュ': 'ju', 'ジョ': 'jo',
            'ビャ': 'bya', 'ビュ': 'byu', 'ビョ': 'byo',
            'ピャ': 'pya', 'ピュ': 'pyu', 'ピョ': 'pyo',
            'ー': '',  # Long vowel mark
            
            # Common kanji words - COMPREHENSIVE
            '練習': 'renshuu', '味見': 'ajimi',
            '一': 'ichi', '二': 'ni', '三': 'san', '四': 'shi', '五': 'go',
            '六': 'roku', '七': 'nana', 'シチ': 'shichi', '八': 'hachi', '九': 'kyuu', '十': 'juu',
            '百': 'hyaku', '千': 'sen', '万': 'man',
            '学': 'gaku', '羊': 'hitsuji', '修': 'shuu', '業': 'gyou',
            '者': 'sha', '山': 'yama', 'やま': 'yama', '後': 'ato', 'のち': 'nochi', '遺産': 'isan',
            '収録': 'shuuroku', '日': 'hi', 'にち': 'nichi', '楽': 'raku', 'がく': 'gaku', '連中': 'renjuu',
            '非': 'hi', '通知': 'tsuuchi', '行': 'kou', 'いく': 'iku', '急落': 'kyuuraku',
            '秋': 'aki', '漁': 'ryou', '救急': 'kyuukyuu', '供給': 'kyoukyuu',
            '生': 'sei', 'なま': 'nama', '要求': 'youkyuu', '空港': 'kuukou', '配置': 'haichi',
            '屋': 'ya', '入': 'nyuu', 'いり': 'iri', 'イチゴ': 'ichigo',
            '八十': 'hachijuu', '四': 'yon', 'よん': 'yon',
            # Numbers as they appear
            '0': 'zero', '1': 'ichi', '2': 'ni', '3': 'san', '4': 'yon', '5': 'go',
            '6': 'roku', '7': 'nana', '8': 'hachi', '9': 'kyuu',
            # Common particles kept as-is
            'を': 'o', 'は': 'wa', 'が': 'ga', 'の': 'no', 'に': 'ni', 'で': 'de',
            'と': 'to', 'や': 'ya', 'か': 'ka', 'ね': 'ne', 'よ': 'yo',
        }
        
        # Try full phrase first (no spaces)
        text_no_space = text.replace(' ', '').replace('　', '')
        if text_no_space in kana_to_romaji:
            return kana_to_romaji[text_no_space]
        
        # Convert character by character for mixed input
        result = []
        i = 0
        while i < len(text):
            # Try 2-character combos first (きゃ, しゃ, キャ, etc.)
            if i + 1 < len(text) and text[i:i+2] in kana_to_romaji:
                result.append(kana_to_romaji[text[i:i+2]])
                i += 2
            # Try single character
            elif text[i] in kana_to_romaji:
                result.append(kana_to_romaji[text[i]])
                i += 1
            # Keep non-kana characters (spaces, numbers, letters)
            elif text[i].isspace():
                i += 1  # Skip spaces
            else:
                result.append(text[i])
                i += 1
        
        text = ''.join(result)
        
        # Romaji normalization
        text = text.replace('wo', 'o').replace(' wa ', ' ha ')
        text = text.replace('uu', 'u').replace('ou', 'o')
        
        # Normalize greeting particles (wa/ha equivalence in common words)
        text = text.replace('konichiwa', 'konnichiha')
        text = text.replace('konbanwa', 'konbanha')
        
        return text
    
    def display_romaji(self, text):
        """Display text fully converted to romaji for user clarity"""
        if not text:
            return "(nothing)"
        
        # Use kakasi for complete conversion
        romaji = self.japanese_to_romaji(text)
        
        # If conversion failed and we still have non-ASCII, show as is
        if not romaji or romaji == text:
            # Try normalize as fallback
            romaji = self.normalize(text)
        
        return romaji if romaji else text
    
    def get_expected_pitch_pattern(self, text):
        """Get expected pitch pattern for common Japanese words/phrases"""
        # Japanese pitch accent patterns
        # ↑ = rising, ↓ = falling, → = flat
        pitch_patterns = {
            # Greetings - typically start high, drop
            'konnichiha': '↓↓',
            'konnichiwa': '↓↓',
            'arigato': '→↓',
            'arigatou': '→↓',
            'sayonara': '→↓',
            'ohayo': '↑↓',
            'ohayou': '↑↓',
            'oyasumi': '→↓',
            'sumimasen': '↑→',
            'gomennasai': '→↓',
            
            # Numbers - mostly flat with final drop
            'ichi': '→→', 'ni': '→→', 'san': '→→',
            'shi': '→→', 'go': '→→', 'roku': '→→',
            'nana': '→→', 'hachi': '→↓', 'kyuu': '→→', 'juu': '→→',
            
            # Common words
            'desu': '→↓',
            'masu': '→↓',
            'genki': '↑→',
            'hai': '→↓',
            'iie': '→↓',
        }
        
        text = text.lower().strip()
        return pitch_patterns.get(text, None)
    
    def phonetic_similarity(self, str1, str2):
        """Calculate phonetic similarity - Japanese sounds that are similar"""
        str1 = str1.lower()
        str2 = str2.lower()
        
        # Special case: Common greetings with は particle (wa/ha confusion)
        # These should be treated as 100% identical
        greeting_pairs = [
            ('konnichiwa', 'konichiha', 'konnichiha'),
            ('konbanwa', 'konbanha', 'konbanha'),
            ('ohayo', 'ohayou'),
            ('arigato', 'arigatou'),
        ]
        
        for pair in greeting_pairs:
            if str1 in pair and str2 in pair:
                return 1.0  # Perfect match
        
        # Common Japanese phonetic equivalences
        equivalents = [
            ['tsu', 'tu', 'zu'],
            ['shi', 'si'],
            ['chi', 'ti'],
            ['fu', 'hu'],
            ['ji', 'zi', 'di'],
            ['n', 'nn', 'nnn'],
            ['uu', 'u'],
            ['ou', 'o'],
            ['ei', 'e'],
            ['wo', 'o'],
            ['r', 'l'],    # R/L confusion common in Japanese
        ]
        
        # Create phonetically normalized versions
        def normalize_phonetic(text):
            # First handle wa/ha at end of greetings (konnichiwa vs konnichiha)
            text = text.replace('ichiwa', 'ichiha')
            text = text.replace('banwa', 'banha')
            
            for group in equivalents:
                for variant in group[1:]:
                    text = text.replace(variant, group[0])
            return text
        
        norm1 = normalize_phonetic(str1)
        norm2 = normalize_phonetic(str2)
        
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    def calculate_score(self, said, correct, pitch_pattern=None):
        """Calculate match score with pitch/inflection analysis"""
        if not said:
            return 0
        
        # Convert Japanese text to romaji for comparison
        said_norm = self.japanese_to_romaji(said)
        correct_norm = self.normalize(correct['romaji'])
        
        # DEBUG: Show what we're comparing
        print(f"[DEBUG] You said: '{said_norm}'")
        print(f"[DEBUG] Expected: '{correct_norm}'")
        
        # Exact match
        if said_norm == correct_norm:
            return 100
        
        # STRICT VALIDATION: Reject completely different words
        # Check basic sequence similarity FIRST - this catches completely wrong words
        quick_check = SequenceMatcher(None, said_norm, correct_norm).ratio()
        if quick_check < 0.4:  # Raised from 0.3 to be stricter
            print(f"{Fore.RED}[VALIDATION] Words too different - {quick_check:.0%} similarity{Style.RESET_ALL}")
            return int(quick_check * 100)
        
        # Length check - if lengths are drastically different, it's wrong
        len_diff_ratio = abs(len(said_norm) - len(correct_norm)) / max(len(said_norm), len(correct_norm), 1)
        if len_diff_ratio > 0.6:  # If one word is 60%+ different in length
            print(f"{Fore.RED}[VALIDATION] Length mismatch - said {len(said_norm)} chars, expected {len(correct_norm)}{Style.RESET_ALL}")
            # Cap score based on how different the lengths are
            length_penalty_score = int((1 - len_diff_ratio) * 50)  # Max 40% if 60% different
            return min(int(quick_check * 100), length_penalty_score)
        
        # Try multiple comparison methods for better accuracy
        scores = []
        
        # 1. Basic fuzzy match
        basic_ratio = SequenceMatcher(None, said_norm, correct_norm).ratio()
        scores.append(basic_ratio * 100)
        
        # 2. Phonetic similarity (accounts for tsu/zu, shi/si, etc.)
        phonetic_score = self.phonetic_similarity(said_norm, correct_norm) * 100
        scores.append(phonetic_score)
        print(f"[PHONETIC] Sound match: {phonetic_score:.0f}%")
        
        # Special handling for R/L confusion (common in Japanese)
        if 'r' in correct_norm or 'l' in correct_norm:
            said_rl = said_norm.replace('r', 'R').replace('l', 'R')
            correct_rl = correct_norm.replace('r', 'R').replace('l', 'R')
            rl_score = SequenceMatcher(None, said_rl, correct_rl).ratio() * 95
            scores.append(rl_score)
        
        # 3. Check if correct answer is contained in what user said
        # STRICT: Must be at least 3 characters or 60% of word length to count
        min_len_for_contain = max(3, int(len(correct_norm) * 0.6))
        if len(correct_norm) >= min_len_for_contain and correct_norm in said_norm:
            scores.append(95)
        
        # 4. Check if user said most of the correct answer
        # STRICT: Only if they said at least 70% of the correct length
        if len(said_norm) >= len(correct_norm) * 0.7 and said_norm in correct_norm:
            percentage = len(said_norm) / len(correct_norm)
            scores.append(percentage * 90)
        
        # 5. Partial match - check if significant portion matches IN CORRECT POSITION
        if len(said_norm) >= len(correct_norm) * 0.6:
            # Check POSITIONAL overlap - not just character presence
            position_matches = 0
            for i, char in enumerate(correct_norm):
                if i < len(said_norm) and said_norm[i] == char:
                    position_matches += 1
            
            if len(correct_norm) > 0:
                # Require at least 50% positional match for any credit
                position_ratio = position_matches / len(correct_norm)
                if position_ratio >= 0.5:
                    overlap_score = position_ratio * 70  # Reduced from 75
                    scores.append(overlap_score)
        
        # 6. Word-by-word comparison (for phrases)
        said_words = said_norm.split()
        correct_words = correct_norm.split()
        if len(correct_words) > 1:
            matching_words = sum(1 for w in correct_words if any(w in sw or sw in w for sw in said_words))
            if len(correct_words) > 0:
                word_score = (matching_words / len(correct_words)) * 85
                scores.append(word_score)
        
        # 7. Starting sound match (first 50% of word)
        half_len = max(2, len(correct_norm) // 2)
        if len(said_norm) >= half_len:
            start_match = SequenceMatcher(None, said_norm[:half_len], correct_norm[:half_len]).ratio()
            scores.append(start_match * 70)
        
        # Get base score
        base_score = int(max(scores)) if scores else 0
        
        # Length difference penalty
        len_diff = abs(len(said_norm) - len(correct_norm))
        max_len = max(len(said_norm), len(correct_norm))
        if max_len > 0:
            len_ratio = len_diff / max_len
            # If lengths differ by more than 50%, apply penalty
            if len_ratio > 0.5:
                penalty = int(len_ratio * 20)  # Up to -20 points
                base_score = max(0, base_score - penalty)
                print(f"[LENGTH] Words very different lengths - penalty: -{penalty}")
        
        # Phonetic bonus - if phonetically similar, boost score
        if base_score >= 40 and base_score < 70:
            phonetic = self.phonetic_similarity(said_norm, correct_norm)
            if phonetic > 0.85:
                phonetic_boost = 15
                base_score = min(100, base_score + phonetic_boost)
                print(f"[PHONETIC] Close pronunciation! +{phonetic_boost}")
        
        # Pitch/inflection bonus (up to +15 points)
        pitch_bonus = 0
        if pitch_pattern and base_score >= 40:
            expected_pattern = self.get_expected_pitch_pattern(correct_norm)
            
            if expected_pattern and pitch_pattern:
                print(f"[PITCH] Your tone: {pitch_pattern} | Expected: {expected_pattern}")
                
                if pitch_pattern == expected_pattern:
                    pitch_bonus = 15
                    print("[PITCH] Perfect inflection! +15")
                elif pitch_pattern[0] == expected_pattern[0]:  # At least start matches
                    pitch_bonus = 8
                    print("[PITCH] Good start inflection! +8")
                else:
                    print("[PITCH] Inflection differs - practice tone")
            elif expected_pattern:
                print(f"[PITCH] Expected pattern: {expected_pattern}")
        
        final_score = min(100, base_score + pitch_bonus)
        return final_score
    
    def practice(self, category):
        """Voice practice session - IMPROVED"""
        print('\n' + '='*70)
        print('[MIC] VOICE PRACTICE - SPEAK THE JAPANESE!')
        print('='*70)
        print('\nHow it works:')
        print('   1. [LISTEN] Listen to Japanese pronunciation')
        print('   2. [MIC] Auto-recording (5 seconds) - SPEAK NOW!')
        print('   3. [SCORE] Get instant feedback')
        print('\n[TIP] Speak CLEARLY and LOUDLY for best recognition')
        print('='*70)
        
        session_score = 0
        session_attempts = 0
        
        while True:
            # Select a valid item with proper romaji
            max_tries = 10
            for _ in range(max_tries):
                item = random.choice(self.content[category])
                # Validate: must have all required fields and romaji must be valid
                if (item.get('romaji') and 
                    item.get('japanese') and 
                    item.get('english') and
                    len(item['romaji']) > 0 and
                    not any(char.isdigit() for char in item['romaji'])):
                    break
            else:
                print("[ERROR] Could not find valid vocabulary item")
                return session_score, session_attempts
            
            print('\n' + '─'*70)
            print(f'{Fore.MAGENTA}{Style.BRIGHT}[JAPANESE]{Style.RESET_ALL} {Fore.WHITE}{item["japanese"]}{Style.RESET_ALL}')
            print(f'{Fore.CYAN}{Style.BRIGHT}[ENGLISH]{Style.RESET_ALL}  {Fore.WHITE}"{item["english"]}"{Style.RESET_ALL}')
            print(f'\n{Back.GREEN}{Fore.BLACK}{Style.BRIGHT} >>> SAY THIS: {item["romaji"]} <<< {Style.RESET_ALL}')
            print(f'{Fore.WHITE}    (pronunciation of: {item["japanese"]}){Style.RESET_ALL}')
            
            print('\n[AUDIO] Listen...')
            # CRITICAL: Use ROMAJI which gets converted to hiragana for accurate TTS
            # This ensures TTS says exactly what the user should say
            self.speak(item['romaji'])  # Converts to hiragana internally
            time.sleep(0.2)  # Brief pause only
            
            # Voice input - 3 attempts
            attempts = 0
            while attempts < 3:
                if attempts > 0:  # Only show attempt number after first try
                    print(f'\n[ATTEMPT {attempts + 1}/3]')
                
                if not self.voice or (not self.voice.whisper_model and not self.voice.vosk_model):
                    print("[X] Voice not available")
                    return session_score, session_attempts
                
                # NO COUNTDOWN - start immediately
                user_speech = self.voice.listen()
                
                if not user_speech:
                    print("[X] No speech - try again (speak LOUDER!)")
                    attempts += 1
                    time.sleep(0.5)  # Reduced delay
                    continue
                
                # Convert to romaji for clear comparison
                user_romaji = self.display_romaji(user_speech)
                correct_romaji = item['romaji']
                
                print(f"\n{Back.BLUE}{Fore.WHITE}{'═'*60}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}{Style.BRIGHT}[YOU SAID]{Style.RESET_ALL}   {Fore.YELLOW}{user_romaji}{Style.RESET_ALL}")
                print(f"{Fore.GREEN}{Style.BRIGHT}[CORRECT]{Style.RESET_ALL}    {Fore.WHITE}{correct_romaji}{Style.RESET_ALL}")
                print(f"{Fore.MAGENTA}[MEANING]{Style.RESET_ALL}    {Fore.WHITE}{item['english']}{Style.RESET_ALL}")
                print(f"{Back.BLUE}{Fore.WHITE}{'═'*60}{Style.RESET_ALL}\n")
                
                # Score it with pitch analysis
                pitch_pattern = None
                if self.voice and hasattr(self.voice, 'last_pitch_pattern'):
                    pitch_pattern = self.voice.last_pitch_pattern
                
                score = self.calculate_score(user_speech, item, pitch_pattern)
                session_attempts += 1
                self.attempts += 1
                attempts += 1
                
                if score >= 70:
                    print(f"{Back.GREEN}{Fore.BLACK} ✓ CORRECT! {score}% {Style.RESET_ALL} {Fore.GREEN}Next word!{Style.RESET_ALL}")
                    session_score += 1
                    self.score += 1
                    self.streak += 1
                    self.best_streak = max(self.best_streak, self.streak)
                    time.sleep(0.5)  # Brief moment to see success
                    break
                elif score >= 55:
                    print(f"{Back.YELLOW}{Fore.BLACK} ~ VERY CLOSE! {score}% {Style.RESET_ALL} {Fore.YELLOW}Try once more{Style.RESET_ALL}")
                    self.speak(item['romaji'])  # Romaji -> hiragana
                    time.sleep(1.5)  # Enough time to see feedback and hear audio
                elif score >= 40:
                    print(f"{Back.YELLOW}{Fore.BLACK} ! Not quite... {score}% {Style.RESET_ALL} {Fore.YELLOW}Listen again{Style.RESET_ALL}")
                    self.speak(item['romaji'])  # Romaji -> hiragana
                    time.sleep(1.5)  # Enough time to process
                else:
                    print(f"{Back.RED}{Fore.WHITE} ✗ INCORRECT ({score}%) {Style.RESET_ALL} {Fore.RED}Listen carefully{Style.RESET_ALL}")
                    self.speak(item['romaji'])  # Romaji -> hiragana
                    time.sleep(2)  # Longer pause to really hear it
                
                if attempts >= 3:
                    print('[SKIP] Moving to next word... (you\'ll see this again)')
                    self.streak = 0
                    time.sleep(1.5)  # Time to see the skip message
                    break
            
            print(f'\n[SCORE] {session_score}/{session_attempts} | [STREAK] {self.streak}')
            time.sleep(0.5)  # Brief pause to see score
    
    def run(self):
        """Main loop - fully voice controlled"""
        print('\n' + '='*70)
        print('   JAPANESE SPEAKING TRAINER - FULL VOICE MODE')
        print('='*70)
        print(f'Python {sys.version_info.major}.{sys.version_info.minor}')
        if self.voice and self.voice.whisper_model:
            print(f'Voice: [OK] Whisper (MAXIMUM ACCURACY)')
        elif self.voice and self.voice.vosk_model:
            print(f'Voice: [OK] Vosk (fallback)')
        else:
            print(f'Voice: [X] Not available')
        print(f'Audio: {"[OK]" if self.use_gtts else "[X]"}')
        print('\n[INFO] Voice-controlled practice')
        print('[EXIT] Press Ctrl+C to stop')
        print('='*70)
        
        # Start continuous practice
        print('\n[START] Beginning continuous practice...')
        print('        Words first, then phrases!')
        time.sleep(2)
        
        while True:
            # Smart progression
            if self.score < 5:
                category = 'words'
                print(f'\n[LEVEL] WORDS (beginner)')
            elif self.score < 10:
                category = 'phrases' if random.random() < 0.5 else 'words'
                print(f'\n[LEVEL] {category.upper()} (intermediate)')
            else:
                category = random.choice(['words', 'phrases'])
                print(f'\n[LEVEL] {category.upper()} (advanced)')
            
            time.sleep(1)
            self.practice(category)

if __name__ == '__main__':
    try:
        trainer = JapaneseSpeakingTrainer()
        trainer.run()
    except KeyboardInterrupt:
        print('\n\n[EXIT] Goodbye! Keep practicing!')
    except Exception as e:
        print(f'\n[ERROR] {e}')
        import traceback
        traceback.print_exc()
