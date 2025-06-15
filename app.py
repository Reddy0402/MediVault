from flask import Flask, request, jsonify, session
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import jwt
import datetime
from functools import wraps

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key' # IMPORTANT: Change this to a strong secret key in production
CORS(app)

# Helper function to get database connection
def get_db_connection():
    conn = sqlite3.connect('medical_records.db')
    conn.row_factory = sqlite3.Row # Allows access to columns by name
    return conn

# Database initialization
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            phone TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Medical records table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS medical_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            record_type TEXT,
            diagnosis TEXT,
            prescription TEXT,
            doctor_name TEXT,
            hospital_name TEXT,
            date_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            file_path TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# --- Authentication Routes ---

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    phone = data.get('phone')

    if not all([username, email, password, phone]):
        return jsonify({'message': 'Missing required fields'}), 400

    hashed_password = generate_password_hash(password)

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, email, password_hash, phone) VALUES (?, ?, ?, ?)",
                       (username, email, hashed_password, phone))
        conn.commit()
        return jsonify({'message': 'User registered successfully!'}), 201
    except sqlite3.IntegrityError:
        return jsonify({'message': 'Username, email or phone already exists'}), 409
    finally:
        conn.close()

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    phone = data.get('phone')
    password = data.get('password')

    if not all([phone, password]):
        return jsonify({'message': 'Missing phone or password'}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    user = cursor.execute("SELECT * FROM users WHERE phone = ?", (phone,)).fetchone()
    conn.close()

    if not user or not check_password_hash(user['password_hash'], password):
        return jsonify({'message': 'Invalid phone or password'}), 401

    # Generate JWT token
    token = jwt.encode({
        'user_id': user['id'],
        'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=30)
    }, app.config['SECRET_KEY'], algorithm="HS256")

    return jsonify({'token': token, 'message': 'Logged in successfully!'}), 200

@app.route('/')
def home():
    return "Hello, MediVault!"

if __name__ == '__main__':
    init_db()
    print("Attempting to run Flask app...")
    app.run(debug=True)