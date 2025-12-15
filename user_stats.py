"""
User Performance Tracking System
Stores user performance data in SQLite database
"""
import sqlite3
import json
from datetime import datetime
from pathlib import Path

class UserStats:
    def __init__(self, db_path='user_performance.db'):
        """Initialize database connection"""
        self.db_path = db_path
        self.conn = None
        self.init_database()
    
    def init_database(self):
        """Create database tables if they don't exist"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.conn.cursor()
        
        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,
                total_attempts INTEGER DEFAULT 0,
                correct_attempts INTEGER DEFAULT 0,
                accuracy REAL DEFAULT 0.0
            )
        ''')
        
        # Individual attempts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attempts (
                attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                word_japanese TEXT,
                word_romaji TEXT,
                word_english TEXT,
                user_said TEXT,
                score INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                passed BOOLEAN,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        ''')
        
        # Word statistics (aggregate performance per word)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS word_stats (
                word_romaji TEXT PRIMARY KEY,
                word_japanese TEXT,
                word_english TEXT,
                times_attempted INTEGER DEFAULT 0,
                times_correct INTEGER DEFAULT 0,
                average_score REAL DEFAULT 0.0,
                last_attempted TIMESTAMP,
                difficulty_rating REAL DEFAULT 50.0
            )
        ''')
        
        # User preferences
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS preferences (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        
        self.conn.commit()
        print("[DATABASE] User statistics database initialized")
    
    def start_session(self):
        """Start a new practice session"""
        cursor = self.conn.cursor()
        cursor.execute('INSERT INTO sessions DEFAULT VALUES')
        self.conn.commit()
        session_id = cursor.lastrowid
        print(f"[SESSION] Started session #{session_id}")
        return session_id
    
    def end_session(self, session_id):
        """End a practice session and calculate final stats"""
        cursor = self.conn.cursor()
        
        # Get session stats
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) as correct
            FROM attempts
            WHERE session_id = ?
        ''', (session_id,))
        
        total, correct = cursor.fetchone()
        accuracy = (correct / total * 100) if total > 0 else 0
        
        cursor.execute('''
            UPDATE sessions 
            SET end_time = CURRENT_TIMESTAMP,
                total_attempts = ?,
                correct_attempts = ?,
                accuracy = ?
            WHERE session_id = ?
        ''', (total, correct, accuracy, session_id))
        
        self.conn.commit()
        print(f"[SESSION] Ended session #{session_id}: {correct}/{total} correct ({accuracy:.1f}%)")
        return {'total': total, 'correct': correct, 'accuracy': accuracy}
    
    def record_attempt(self, session_id, word_item, user_said, score, passed):
        """Record a single word attempt"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO attempts (
                session_id, word_japanese, word_romaji, word_english,
                user_said, score, passed
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            word_item.get('japanese', ''),
            word_item.get('romaji', ''),
            word_item.get('english', ''),
            user_said,
            score,
            passed
        ))
        
        # Update word statistics
        cursor.execute('''
            INSERT INTO word_stats (
                word_romaji, word_japanese, word_english,
                times_attempted, times_correct, average_score, last_attempted
            ) VALUES (?, ?, ?, 1, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(word_romaji) DO UPDATE SET
                times_attempted = times_attempted + 1,
                times_correct = times_correct + ?,
                average_score = (average_score * times_attempted + ?) / (times_attempted + 1),
                last_attempted = CURRENT_TIMESTAMP
        ''', (
            word_item.get('romaji', ''),
            word_item.get('japanese', ''),
            word_item.get('english', ''),
            1 if passed else 0,
            score,
            1 if passed else 0,
            score
        ))
        
        self.conn.commit()
    
    def get_word_difficulty(self, word_romaji):
        """Get difficulty rating for a word (0-100, higher = harder)"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT 
                times_attempted,
                times_correct,
                average_score
            FROM word_stats
            WHERE word_romaji = ?
        ''', (word_romaji,))
        
        result = cursor.fetchone()
        if not result:
            return 50.0  # Default medium difficulty
        
        attempts, correct, avg_score = result
        if attempts == 0:
            return 50.0
        
        # Calculate difficulty: lower success rate = higher difficulty
        success_rate = (correct / attempts) * 100
        difficulty = 100 - success_rate
        return difficulty
    
    def get_worst_words(self, limit=10):
        """Get words the user struggles with most"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT 
                word_romaji,
                word_japanese,
                word_english,
                times_attempted,
                times_correct,
                CAST(times_correct AS REAL) / times_attempted * 100 as success_rate
            FROM word_stats
            WHERE times_attempted >= 3
            ORDER BY success_rate ASC, times_attempted DESC
            LIMIT ?
        ''', (limit,))
        
        return cursor.fetchall()
    
    def get_session_history(self, limit=10):
        """Get recent session summaries"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT 
                session_id,
                start_time,
                end_time,
                total_attempts,
                correct_attempts,
                accuracy
            FROM sessions
            WHERE end_time IS NOT NULL
            ORDER BY start_time DESC
            LIMIT ?
        ''', (limit,))
        
        return cursor.fetchall()
    
    def get_overall_stats(self):
        """Get overall user statistics"""
        cursor = self.conn.cursor()
        
        # Total stats
        cursor.execute('''
            SELECT 
                COUNT(*) as total_sessions,
                SUM(total_attempts) as total_attempts,
                SUM(correct_attempts) as correct_attempts,
                AVG(accuracy) as avg_accuracy
            FROM sessions
            WHERE end_time IS NOT NULL
        ''')
        
        stats = cursor.fetchone()
        
        # Unique words practiced
        cursor.execute('SELECT COUNT(DISTINCT word_romaji) FROM word_stats')
        unique_words = cursor.fetchone()[0]
        
        return {
            'total_sessions': stats[0] or 0,
            'total_attempts': stats[1] or 0,
            'correct_attempts': stats[2] or 0,
            'avg_accuracy': stats[3] or 0.0,
            'unique_words_practiced': unique_words
        }
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("[DATABASE] Connection closed")
