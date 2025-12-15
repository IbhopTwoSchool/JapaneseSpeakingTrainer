"""
Japanese Speaking Trainer - GUI Version
Real-time visual feedback for pronunciation training
"""
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import queue
from speak import JapaneseSpeakingTrainer
import time
import random
import math
from user_stats import UserStats

class SpeakingTrainerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Japanese Speaking Trainer - Live Feedback")
        self.root.geometry("1000x900")  # Increased from 900x700 to fit all elements
        self.root.minsize(900, 800)  # Set minimum size to prevent elements from being cut off
        self.root.configure(bg='#1e1e1e')
        
        # Message queue for thread-safe GUI updates
        self.message_queue = queue.Queue()
        
        # Initialize trainer
        self.trainer = JapaneseSpeakingTrainer()
        self.current_item = None
        self.is_recording = False
        
        # Waveform animation state
        self.waveform_active = False
        self.waveform_color = '#4CAF50'
        self.wave_bars = []
        
        # Session tracking
        self.user_stats = UserStats()
        self.session_id = None
        self.used_words_this_session = set()
        self.available_words = []
        
        self.setup_gui()
        self.root.bind('<Configure>', self.on_window_resize)  # Handle window resize
        self.process_queue()
        self.animate_waveform()
        
    def setup_gui(self):
        """Setup the GUI layout"""
        
        # Header
        header = tk.Frame(self.root, bg='#2d2d2d', height=60)
        header.pack(fill=tk.X, padx=10, pady=5)
        
        title = tk.Label(header, text="🎌 Japanese Speaking Trainer", 
                        font=('Arial', 20, 'bold'), bg='#2d2d2d', fg='#4CAF50')
        title.pack(pady=10)
        
        # Main content area
        content = tk.Frame(self.root, bg='#1e1e1e')
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Word display section
        self.word_frame = tk.LabelFrame(content, text="Current Word", 
                                        font=('Arial', 12, 'bold'),
                                        bg='#2d2d2d', fg='white', 
                                        relief=tk.GROOVE, borderwidth=2)
        self.word_frame.pack(fill=tk.X, pady=5)
        
        # Japanese characters (Kanji)
        self.japanese_label = tk.Label(self.word_frame, text="準備中...", 
                                      font=('MS Gothic', 48, 'bold'),
                                      bg='#2d2d2d', fg='#BB86FC')
        self.japanese_label.pack(pady=5)
        
        # Hiragana representation
        self.hiragana_label = tk.Label(self.word_frame, text="",
                                      font=('MS Gothic', 20),
                                      bg='#2d2d2d', fg='#64B5F6')
        self.hiragana_label.pack()
        
        # Romaji (what to say)
        self.romaji_frame = tk.Frame(self.word_frame, bg='#4CAF50', relief=tk.RAISED, borderwidth=3)
        self.romaji_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.romaji_label = tk.Label(self.romaji_frame, text="Say This: ...", 
                                     font=('Arial', 32, 'bold'),
                                     bg='#4CAF50', fg='black')
        self.romaji_label.pack(pady=10)
        
        # English meaning
        self.english_label = tk.Label(self.word_frame, text="Meaning: ...", 
                                     font=('Arial', 16),
                                     bg='#2d2d2d', fg='#03DAC6')
        self.english_label.pack(pady=5)
        
        # Breakdown explanation
        self.breakdown_label = tk.Label(self.word_frame, text="",
                                       font=('Arial', 12),
                                       bg='#2d2d2d', fg='#FFA726',
                                       wraplength=900, justify='left')  # Increased wraplength
        self.breakdown_label.pack(pady=5, padx=10, fill=tk.X)
        
        # Recording status
        self.recording_frame = tk.Frame(content, bg='#2d2d2d', height=80)
        self.recording_frame.pack(fill=tk.X, pady=10)
        
        self.recording_label = tk.Label(self.recording_frame, text="⚪ Ready", 
                                       font=('Arial', 24, 'bold'),
                                       bg='#2d2d2d', fg='#888888')
        self.recording_label.pack(pady=5)
        
        # Waveform visualization
        self.waveform_canvas = tk.Canvas(self.recording_frame, bg='#1e1e1e', 
                                         height=60, highlightthickness=0)
        self.waveform_canvas.pack(fill=tk.X, padx=20, pady=5)
        self.create_waveform_bars()
        
        # Results section
        self.results_frame = tk.LabelFrame(content, text="Recognition Results", 
                                          font=('Arial', 12, 'bold'),
                                          bg='#2d2d2d', fg='white',
                                          relief=tk.GROOVE, borderwidth=2)
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # What AI heard
        heard_frame = tk.Frame(self.results_frame, bg='#2d2d2d')
        heard_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(heard_frame, text="AI Heard:", font=('Arial', 12, 'bold'),
                bg='#2d2d2d', fg='#FFA726').pack(anchor='w')
        
        self.heard_label = tk.Label(heard_frame, text="...", 
                                   font=('Arial', 18),
                                   bg='#2d2d2d', fg='#FFEB3B',
                                   wraplength=900, justify='left')  # Increased wraplength
        self.heard_label.pack(anchor='w', pady=2, fill=tk.X)
        
        # Expected vs Actual comparison
        compare_frame = tk.Frame(self.results_frame, bg='#2d2d2d')
        compare_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(compare_frame, text="You Said:", font=('Arial', 12, 'bold'),
                bg='#2d2d2d', fg='#64B5F6').pack(anchor='w')
        self.you_said_label = tk.Label(compare_frame, text="...", 
                                      font=('Arial', 16),
                                      bg='#2d2d2d', fg='white')
        self.you_said_label.pack(anchor='w', pady=2)
        
        tk.Label(compare_frame, text="Expected:", font=('Arial', 12, 'bold'),
                bg='#2d2d2d', fg='#81C784').pack(anchor='w')
        self.expected_label = tk.Label(compare_frame, text="...", 
                                      font=('Arial', 16),
                                      bg='#2d2d2d', fg='white')
        self.expected_label.pack(anchor='w', pady=2)
        
        # Score display
        self.score_label = tk.Label(self.results_frame, text="Score: ...", 
                                   font=('Arial', 20, 'bold'),
                                   bg='#2d2d2d', fg='white')
        self.score_label.pack(pady=10)
        
        # Status bar
        status_frame = tk.Frame(self.root, bg='#2d2d2d', height=40)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = tk.Label(status_frame, text="Ready to start", 
                                    font=('Arial', 10),
                                    bg='#2d2d2d', fg='#888888')
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        self.streak_label = tk.Label(status_frame, text="Streak: 0 | Score: 0/0", 
                                    font=('Arial', 10, 'bold'),
                                    bg='#2d2d2d', fg='#4CAF50')
        self.streak_label.pack(side=tk.RIGHT, padx=10)
        
        # Control buttons
        button_frame = tk.Frame(content, bg='#1e1e1e')
        button_frame.pack(fill=tk.X, pady=10)
        
        self.start_button = tk.Button(button_frame, text="▶ Start Training", 
                                      font=('Arial', 14, 'bold'),
                                      bg='#4CAF50', fg='white',
                                      command=self.start_training,
                                      width=20, height=2)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.skip_button = tk.Button(button_frame, text="⏭ Skip Word", 
                                     font=('Arial', 14, 'bold'),
                                     bg='#FF9800', fg='white',
                                     command=self.skip_word,
                                     width=15, height=2,
                                     state=tk.DISABLED)
        self.skip_button.pack(side=tk.LEFT, padx=5)
    
    def create_waveform_bars(self):
        """Create waveform bar elements"""
        self.wave_bars = []
        self.num_bars = 40
        self.update_waveform_bars()
    
    def update_waveform_bars(self):
        """Update waveform bar positions based on canvas size"""
        # Clear existing bars
        self.waveform_canvas.delete('all')
        self.wave_bars = []
        
        # Get current canvas width
        canvas_width = self.waveform_canvas.winfo_width()
        if canvas_width <= 1:  # Canvas not yet rendered
            canvas_width = 900
        
        bar_width = max(2, (canvas_width / self.num_bars) - 2)
        
        for i in range(self.num_bars):
            x = i * (bar_width + 2) + bar_width/2
            bar = self.waveform_canvas.create_rectangle(
                x, 30, x + bar_width, 30,
                fill='#4CAF50', outline=''
            )
            self.wave_bars.append(bar)
    
    def on_window_resize(self, event):
        """Handle window resize events - update wraplenghts and waveform"""
        if event.widget == self.root:
            # Update text wraplengths based on new window width
            new_width = max(800, event.width - 100)
            
            self.breakdown_label.config(wraplength=new_width)
            self.heard_label.config(wraplength=new_width)
            
            # Redraw waveform bars to fit new width
            self.root.after(100, self.update_waveform_bars)  # Delay to let canvas resize
    
    def update_waveform_levels(self, levels):
        """Update waveform bars with real audio levels"""
        if len(levels) != len(self.wave_bars):
            return
        
        for i, (bar, level) in enumerate(zip(self.wave_bars, levels)):
            # Scale level to bar height (max 50px)
            height = max(2, min(50, level))
            
            # Calculate bar position
            x1, y1, x2, y2 = self.waveform_canvas.coords(bar)
            center_y = 30
            new_y1 = center_y - height/2
            new_y2 = center_y + height/2
            
            self.waveform_canvas.coords(bar, x1, new_y1, x2, new_y2)
            self.waveform_canvas.itemconfig(bar, fill=self.waveform_color)
    
    def animate_waveform(self):
        """Animate waveform bars (only used when no real audio data)"""
        # Only animate if active but no recent audio data update
        # This serves as a fallback visual effect
        if not self.waveform_active:
            # Idle state - flat minimal bars
            for bar in self.wave_bars:
                x1, y1, x2, y2 = self.waveform_canvas.coords(bar)
                self.waveform_canvas.coords(bar, x1, 28, x2, 32)
                self.waveform_canvas.itemconfig(bar, fill='#333333')
        
        # Continue animation loop
        self.root.after(50, self.animate_waveform)
    
    def process_queue(self):
        """Process messages from training thread"""
        try:
            while True:
                msg = self.message_queue.get_nowait()
                self.handle_message(msg)
        except queue.Empty:
            pass
        self.root.after(100, self.process_queue)
    
    def handle_message(self, msg):
        """Handle different message types"""
        msg_type = msg.get('type')
        
        if msg_type == 'new_word':
            self.display_word(msg['item'])
        elif msg_type == 'tts_start':
            self.show_tts_speaking()
        elif msg_type == 'tts_end':
            self.stop_waveform()
        elif msg_type == 'recording_start':
            self.show_recording()
        elif msg_type == 'recording_end':
            self.show_processing()
        elif msg_type == 'audio_levels':
            self.update_waveform_levels(msg['levels'])
        elif msg_type == 'recognition':
            self.show_recognition(msg['text'], msg.get('confidence', 0))
        elif msg_type == 'result':
            self.show_result(msg)
        elif msg_type == 'status':
            self.update_status(msg['text'])
        elif msg_type == 'score':
            self.update_score(msg['score'], msg['total'], msg['streak'])
    
    def display_word(self, item):
        """Display the current word with all representations"""
        self.current_item = item
        
        # Show main Japanese (kanji/kana)
        self.japanese_label.config(text=item['japanese'])
        
        # Show hiragana reading if available (from breakdown or conversion)
        hiragana_text = ""
        if 'hiragana' in item and item['hiragana']:
            hiragana_text = f"読み: {item['hiragana']}"
        self.hiragana_label.config(text=hiragana_text)
        
        # Show romaji
        self.romaji_label.config(text=f"Say This: {item['romaji']}")
        
        # Show English
        self.english_label.config(text=f"Meaning: {item['english']}")
        
        # Show breakdown if available
        breakdown_text = ""
        if 'breakdown' in item and item['breakdown']:
            breakdown_text = f"📖 {item['breakdown']}"
        self.breakdown_label.config(text=breakdown_text)
        
        # Reset results
        self.heard_label.config(text="Waiting for speech...")
        self.you_said_label.config(text="...")
        self.expected_label.config(text=item['romaji'])
        self.score_label.config(text="Listening...")
    
    def show_tts_speaking(self):
        """Show TTS speaking with waveform"""
        self.waveform_active = True
        self.waveform_color = '#03DAC6'  # Cyan for TTS
        self.recording_label.config(text="🔊 Computer Speaking", fg='#03DAC6')
    
    def stop_waveform(self):
        """Stop waveform animation"""
        self.waveform_active = False
    
    def show_recording(self):
        """Show recording status"""
        self.recording_label.config(text="🔴 RECORDING - SPEAK NOW!", 
                                   bg='#2d2d2d', fg='#FF5252')
        self.recording_frame.config(bg='#FF5252')
        self.is_recording = True
        self.waveform_active = True
        self.waveform_color = '#FF5252'  # Red for recording
    
    def show_processing(self):
        """Show processing status"""
        self.recording_label.config(text="⏳ Processing...", 
                                   fg='#FFA726')
        self.recording_frame.config(bg='#2d2d2d')
        self.is_recording = False
        self.waveform_active = False
    
    def show_recognition(self, text, confidence):
        """Show what AI recognized"""
        self.heard_label.config(text=f"{text} (Confidence: {confidence:.0%})")
        romaji = self.trainer.japanese_to_romaji(text)
        self.you_said_label.config(text=romaji)
    
    def show_result(self, result):
        """Show comparison result"""
        score = result['score']
        
        if score >= 70:
            color = '#4CAF50'  # Green
            message = f"✓ CORRECT! {score}%"
            self.score_label.config(bg='#4CAF50', fg='black')
        elif score >= 55:
            color = '#FFA726'  # Orange
            message = f"~ VERY CLOSE! {score}%"
            self.score_label.config(bg='#FFA726', fg='black')
        elif score >= 40:
            color = '#FF9800'  # Dark orange
            message = f"! Not quite... {score}%"
            self.score_label.config(bg='#FF9800', fg='black')
        else:
            color = '#F44336'  # Red
            message = f"✗ INCORRECT {score}%"
            self.score_label.config(bg='#F44336', fg='white')
        
        self.score_label.config(text=message)
        self.recording_label.config(text="⚪ Ready for next", fg='#888888')
        self.recording_frame.config(bg='#2d2d2d')
        self.waveform_active = False
    
    def update_status(self, text):
        """Update status message"""
        self.status_label.config(text=text)
    
    def update_score(self, score, total, streak):
        """Update score display"""
        self.streak_label.config(text=f"Streak: {streak} | Score: {score}/{total}")
    
    def get_next_word(self):
        """Get next word, avoiding duplicates within session"""
        # Initialize available words if empty
        if not self.available_words:
            self.available_words = self.trainer.content['words'].copy()
            # Filter out already-used words
            self.available_words = [w for w in self.available_words 
                                   if w['romaji'] not in self.used_words_this_session]
            
            # If all words used, reset for new cycle
            if not self.available_words:
                print("[SESSION] Completed full cycle! Resetting words...")
                self.used_words_this_session.clear()
                self.available_words = self.trainer.content['words'].copy()
        
        # Get random word from available pool
        word = random.choice(self.available_words)
        self.available_words.remove(word)
        self.used_words_this_session.add(word['romaji'])
        
        return word
    
    def start_training(self):
        """Start training session in background thread"""
        self.start_button.config(state=tk.DISABLED)
        self.skip_button.config(state=tk.NORMAL)
        
        # Start database session
        self.session_id = self.user_stats.start_session()
        
        threading.Thread(target=self.training_loop, daemon=True).start()
    
    def skip_word(self):
        """Skip current word"""
        self.message_queue.put({'type': 'status', 'text': 'Skipping to next word...'})
        # Signal to skip in training loop (we'll implement this)
    
    def training_loop(self):
        """Main training loop running in background - OPTIMIZED"""
        session_score = 0
        session_attempts = 0
        current_item = None
        
        while True:
            # Get word - use same word if failed last time, otherwise get new unique one
            if current_item is None:
                current_item = self.get_next_word()
            
            item = current_item
            self.message_queue.put({'type': 'new_word', 'item': item})
            time.sleep(0.5)
            
            # Play audio with waveform - simulate TTS levels
            self.message_queue.put({'type': 'tts_start'})
            
            # Start TTS playback
            tts_thread = threading.Thread(target=self.trainer.speak, args=(item['romaji'],), daemon=True)
            tts_thread.start()
            
            # Simulate waveform during TTS (since we can't easily capture pygame audio)
            # Generate random levels that look like speech
            tts_start_time = time.time()
            while tts_thread.is_alive():
                # Generate speech-like waveform levels
                levels = [random.uniform(10, 60) * (0.5 + 0.5 * random.random()) for _ in range(40)]
                self.message_queue.put({'type': 'audio_levels', 'levels': levels})
                time.sleep(0.05)  # Update 20 times per second
            
            self.message_queue.put({'type': 'tts_end'})
            time.sleep(0.3)
            
            # Record with real-time audio callback
            self.message_queue.put({'type': 'recording_start'})
            
            def audio_level_callback(levels):
                """Receive real-time audio levels from microphone"""
                self.message_queue.put({'type': 'audio_levels', 'levels': levels})
            
            user_speech = self.trainer.voice.listen(
                audio_callback=audio_level_callback,
                expected_word=item['romaji']  # Pass expected word for hallucination detection
            )
            self.message_queue.put({'type': 'recording_end'})
            
            if user_speech:
                # Show recognition
                self.message_queue.put({
                    'type': 'recognition',
                    'text': user_speech,
                    'confidence': 0.85  # Placeholder
                })
                
                # Calculate score
                score = self.trainer.calculate_score(user_speech, item, None)
                session_attempts += 1
                passed = score >= 70
                
                # Record to database
                user_said_romaji = self.trainer.japanese_to_romaji(user_speech)
                self.user_stats.record_attempt(
                    self.session_id, item, user_said_romaji, score, passed
                )
                
                if passed:
                    session_score += 1
                    self.trainer.score += 1
                    self.trainer.streak += 1
                    # PASSED - move to next word
                    current_item = None
                    time.sleep(1)
                else:
                    # FAILED - keep same word for retry
                    self.message_queue.put({
                        'type': 'status',
                        'text': 'TRY AGAIN - Same word!'
                    })
                    time.sleep(2)
                
                # Show result
                self.message_queue.put({
                    'type': 'result',
                    'score': score,
                    'user_said': self.trainer.japanese_to_romaji(user_speech),
                    'expected': item['romaji']
                })
                
                # Update score
                self.message_queue.put({
                    'type': 'score',
                    'score': session_score,
                    'total': session_attempts,
                    'streak': self.trainer.streak
                })
            else:
                self.message_queue.put({
                    'type': 'status',
                    'text': 'No speech detected - try again'
                })
                time.sleep(1)


if __name__ == '__main__':
    root = tk.Tk()
    app = SpeakingTrainerGUI(root)
    root.mainloop()
