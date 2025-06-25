import sqlite3

# Database setup
def create_user_table():
    conn = sqlite3.connect("thyroid.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Add sample user
def populate_users():
    conn = sqlite3.connect("thyroid.db")
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO users (username, password) VALUES (?, ?)", 
                   ("test_user", "test123"))  # Sample user
    conn.commit()
    conn.close()

# Get user from database
def get_user(username):
    conn = sqlite3.connect("thyroid.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username=?", (username,))
    user = cursor.fetchone()
    conn.close()
    return user
