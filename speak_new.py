import random
import time
import sys
from difflib import SequenceMatcher

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

class JapaneseSpeakingTrainer:
    def __init__(self):
        if TTS_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 130)
            except:
                self.engine = None
        else:
            self.engine = None
            
        self.score = 0
        self.attempts = 0
        self.streak = 0
        self.user_level = 1
        self.words_mastered = 0
        self.phrases_mastered = 0
        
        self.content = {
            'words': [
                {'japanese': 'こんにちは', 'romaji': 'konnichiwa', 'english': 'hello'},
                {'japanese': 'ありがとう', 'romaji': 'arigatou', 'english': 'thank you'},
                {'japanese': 'さようなら', 'romaji': 'sayounara', 'english': 'goodbye'},
                {'japanese': 'おはよう', 'romaji': 'ohayou', 'english': 'good morning'},
                {'japanese': 'おやすみ', 'romaji': 'oyasumi', 'english': 'good night'},
                {'japanese': 'はい', 'romaji': 'hai', 'english': 'yes'},
                {'japanese': 'いいえ', 'romaji': 'iie', 'english': 'no'},
                {'japanese': 'すみません', 'romaji': 'sumimasen', 'english': 'excuse me'},
                {'japanese': 'ごめんなさい', 'romaji': 'gomennasai', 'english': 'sorry'},
                {'japanese': 'いただきます', 'romaji': 'itadakimasu', 'english': "let's eat"},
            ],
            'phrases': [
                {'japanese': 'お元気ですか', 'romaji': 'ogenki desu ka', 'english': 'how are you'},
                {'japanese': '元気です', 'romaji': 'genki desu', 'english': 'I am fine'},
                {'japanese': 'お名前は何ですか', 'romaji': 'onamae wa nan desu ka', 'english': 'what is your name'},
                {'japanese': '私の名前は', 'romaji': 'watashi no namae wa', 'english': 'my name is'},
                {'japanese': 'どういたしまして', 'romaji': 'douitashimashite', 'english': "you're welcome"},
                {'japanese': 'わかりません', 'romaji': 'wakarimasen', 'english': "I don't understand"},
                {'japanese': 'もう一度お願いします', 'romaji': 'mou ichido onegaishimasu', 'english': 'please say it again'},
                {'japanese': 'お疲れ様でした', 'romaji': 'otsukaresama deshita', 'english': 'good work'},
                {'japanese': 'いってきます', 'romaji': 'ittekimasu', 'english': "I'm leaving"},
                {'japanese': 'いってらっしゃい', 'romaji': 'itterasshai', 'english': 'have a good day'},
            ],
            'sentences': [
                {'japanese': '私は日本語を勉強しています', 'romaji': 'watashi wa nihongo wo benkyou shiteimasu', 'english': 'I am studying Japanese'},
                {'japanese': '日本に行きたいです', 'romaji': 'nihon ni ikitai desu', 'english': 'I want to go to Japan'},
                {'japanese': 'これはいくらですか', 'romaji': 'kore wa ikura desu ka', 'english': 'how much is this'},
                {'japanese': '日本語が少し話せます', 'romaji': 'nihongo ga sukoshi hanasemasu', 'english': 'I can speak a little Japanese'},
                {'japanese': 'トイレはどこですか', 'romaji': 'toire wa doko desu ka', 'english': 'where is the bathroom'},
                {'japanese': '水をください', 'romaji': 'mizu wo kudasai', 'english': 'please give me water'},
                {'japanese': '英語を話せますか', 'romaji': 'eigo wo hanasemasu ka', 'english': 'can you speak English'},
                {'japanese': '今何時ですか', 'romaji': 'ima nanji desu ka', 'english': 'what time is it now'},
                {'japanese': '駅はどこですか', 'romaji': 'eki wa doko desu ka', 'english': 'where is the station'},
                {'japanese': '私は学生です', 'romaji': 'watashi wa gakusei desu', 'english': 'I am a student'},
            ]
        }
        
    def display_menu(self):
        print('\n' + '='*60)
        print('🎌 JAPANESE SPEAKING TRAINER 🎌')
        print('='*60)
        level_name = ['Beginner (Words)', 'Intermediate (Phrases)', 'Advanced (Sentences)'][self.user_level - 1]
        print(f'\n📊 Current Level: {level_name}')
        print(f'✅ Score: {self.score}/{self.attempts} | 🔥 Streak: {self.streak}')
        print(f'📚 Mastered: {self.words_mastered} words, {self.phrases_mastered} phrases')
        print('\n1. Start Progressive Practice (Recommended!)')
        print('2. Practice Words Only')
        print('3. Practice Phrases Only')
        print('4. Practice Sentences Only')
        print('5. View Statistics')
        print('6. Exit')
        print('='*60)
        
    def get_random_item(self, category=None):
        if category:
            return random.choice(self.content[category])
        else:
            all_items = []
            for cat in self.content.values():
                all_items.extend(cat)
            return random.choice(all_items)
    
    def speak_text(self, text):
        if self.engine:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except:
                pass
        print(f'🔊 {text}')
    
    def normalize_text(self, text):
        text = text.lower().replace(' ', '').replace('　', '')
        replacements = {
            'wa': 'ha', 'wo': 'o', 'ー': '', 'っ': '', 'ん': 'n'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
    
    def check_similarity(self, user_input, correct_answer):
        user_normalized = self.normalize_text(user_input)
        correct_normalized = self.normalize_text(correct_answer)
        
        if user_normalized == correct_normalized:
            return 100
        if correct_normalized in user_normalized:
            return 95
        if user_normalized in correct_normalized:
            return 90
        
        ratio = SequenceMatcher(None, user_normalized, correct_normalized).ratio()
        return int(ratio * 100)
    
    def practice_session(self, category=None, progressive=False):
        print('\n' + '='*60)
        if progressive:
            print('📚 PROGRESSIVE PRACTICE - Listen, Learn, Type!')
            print('   Start with words → advance to phrases → master sentences')
        else:
            print('📚 PRACTICE SESSION')
        print('='*60)
        print('\n📖 How it works:')
        print('   1. Listen to the Japanese pronunciation (audio + text)')
        print('   2. Try to say it out loud yourself')
        print('   3. Type the romaji to test your memory')
        print('   4. Get instant feedback!')
        print('\n💡 Commands: "replay" | "skip" | "quit"')
        print('='*60)
        
        session_score = 0
        session_attempts = 0
        session_streak = 0
        
        if progressive:
            if self.user_level == 1:
                current_category = 'words'
            elif self.user_level == 2:
                current_category = 'phrases'
            else:
                current_category = 'sentences'
        else:
            current_category = category
        
        while True:
            if progressive:
                item = self.get_random_item(current_category)
            else:
                item = self.get_random_item(category)
            
            print('\n' + '-'*60)
            if progressive:
                print(f'📊 Level: {current_category.upper()}')
            print(f'\n📝 {item["japanese"]}')
            print(f'🇬🇧 "{item["english"]}"')
            print('\n🔊 Listen carefully...')
            time.sleep(0.5)
            self.speak_text(item['romaji'])
            time.sleep(0.3)
            self.speak_text(item['romaji'])
            
            attempts_for_item = 0
            max_attempts = 3
            
            while attempts_for_item < max_attempts:
                print(f'\n⌨️  Now YOU try! Type the romaji (Attempt {attempts_for_item + 1}/{max_attempts}):')
                user_input = input('➡️  ').strip()
                
                if not user_input:
                    print('❌ No input. Try again!')
                    continue
                
                user_lower = user_input.lower()
                
                if user_lower == 'quit':
                    return session_score, session_attempts
                elif user_lower == 'skip':
                    print('⏭️  Skipped!')
                    session_streak = 0
                    break
                elif user_lower == 'replay':
                    self.speak_text(item['romaji'])
                    continue
                else:
                    similarity_jp = self.check_similarity(user_input, item['japanese'])
                    similarity_roma = self.check_similarity(user_input, item['romaji'])
                    similarity = max(similarity_jp, similarity_roma)
                    
                    session_attempts += 1
                    self.attempts += 1
                    attempts_for_item += 1
                    
                    if similarity >= 85:
                        print(f'✅ PERFECT! {similarity}% match!')
                        print(f'   Correct: {item["romaji"]}')
                        session_score += 1
                        self.score += 1
                        session_streak += 1
                        self.streak = max(self.streak, session_streak)
                        
                        if current_category == 'words':
                            self.words_mastered += 1
                        elif current_category == 'phrases':
                            self.phrases_mastered += 1
                        
                        if progressive:
                            if current_category == 'words' and self.words_mastered >= 5:
                                print('\n🎉 LEVEL UP! Moving to PHRASES! 🎉')
                                current_category = 'phrases'
                                self.user_level = 2
                                time.sleep(2)
                            elif current_category == 'phrases' and self.phrases_mastered >= 5:
                                print('\n🎉 LEVEL UP! Moving to SENTENCES! 🎉')
                                current_category = 'sentences'
                                self.user_level = 3
                                time.sleep(2)
                        break
                    elif similarity >= 70:
                        print(f'👍 Close! {similarity}% - Listen again:')
                        self.speak_text(item['romaji'])
                        print(f'   Correct: {item["romaji"]}')
                        session_streak = 0
                    else:
                        print(f'❌ Not quite ({similarity}%)')
                        print(f'   Correct: {item["romaji"]}')
                        self.speak_text(item['romaji'])
                        session_streak = 0
                        
                        if attempts_for_item >= max_attempts:
                            print('   Moving on...')
                            break
            
            print(f'\n📊 Session: {session_score}/{session_attempts} | 🔥 Streak: {session_streak}')
            
            cont = input('➡️  Continue? (y/n): ').strip().lower()
            if cont != 'y':
                return session_score, session_attempts
    
    def view_statistics(self):
        print('\n' + '='*60)
        print('📊 YOUR PROGRESS')
        print('='*60)
        level_name = ['Beginner (Words)', 'Intermediate (Phrases)', 'Advanced (Sentences)'][self.user_level - 1]
        print(f'\n🎯 Level: {level_name}')
        print(f'✅ Correct: {self.score}')
        print(f'📝 Total Attempts: {self.attempts}')
        print(f'🔥 Best Streak: {self.streak}')
        print(f'📚 Words Mastered: {self.words_mastered}')
        print(f'💬 Phrases Mastered: {self.phrases_mastered}')
        if self.attempts > 0:
            accuracy = (self.score / self.attempts) * 100
            print(f'🎯 Accuracy: {accuracy:.1f}%')
        print('='*60)
    
    def run(self):
        print('\n🎌 JAPANESE SPEAKING TRAINER 🎌')
        print('Learn Japanese pronunciation through listening and typing!')
        print(f'\nPython {sys.version_info.major}.{sys.version_info.minor} | Audio: {"✅" if self.engine else "❌"}')
        
        while True:
            self.display_menu()
            choice = input('\n➡️  Choice (1-6): ').strip()
            
            if choice == '1':
                score, attempts = self.practice_session(None, progressive=True)
                print(f'\n✅ Session complete! {score}/{attempts}')
            elif choice == '2':
                score, attempts = self.practice_session('words')
                print(f'\n✅ Complete! {score}/{attempts}')
            elif choice == '3':
                score, attempts = self.practice_session('phrases')
                print(f'\n✅ Complete! {score}/{attempts}')
            elif choice == '4':
                score, attempts = self.practice_session('sentences')
                print(f'\n✅ Complete! {score}/{attempts}')
            elif choice == '5':
                self.view_statistics()
            elif choice == '6':
                print('\n👋 がんばって! (Ganbatte - Good luck!)')
                break
            else:
                print('❌ Invalid choice (1-6)')
            
            input('\nPress Enter...')

if __name__ == '__main__':
    try:
        trainer = JapaneseSpeakingTrainer()
        trainer.run()
    except KeyboardInterrupt:
        print('\n\n👋 さようなら! (Sayounara!)')
    except Exception as e:
        print(f'\n❌ Error: {e}')
