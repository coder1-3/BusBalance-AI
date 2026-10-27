
from flask import Flask, render_template, request, redirect, session, flash, g
import sqlite3
import os
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'super_secret_transit_key'
DATABASE = 'busbalance.db'

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get("user_id") is None:
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated_function

def role_required(role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if session.get("role") != role:
                flash("Unauthorized access. Please login with correct credentials.", "error")
                return redirect("/")
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def init_db():
    if not os.path.exists(DATABASE):
        with sqlite3.connect(DATABASE) as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT, name TEXT, role TEXT)''')
            c.execute('''CREATE TABLE schedules (id INTEGER PRIMARY KEY, route_name TEXT, departure_time TEXT, capacity INTEGER, booked_seats INTEGER DEFAULT 0, overflow_count INTEGER DEFAULT 0, driver_name TEXT, conductor_name TEXT)''')
            c.execute('''CREATE TABLE bookings (id INTEGER PRIMARY KEY, user_id INTEGER, schedule_id INTEGER, status TEXT)''')
            
            # Dummy data
            c.execute("INSERT INTO users (username, password, name, role) VALUES (?, ?, ?, ?)", ('admin', generate_password_hash('admin123'), 'System Admin', 'staff'))
            c.execute("INSERT INTO users (username, password, name, role) VALUES (?, ?, ?, ?)", ('user', generate_password_hash('user123'), 'Senior Citizen', 'passenger'))
            
            schedules = [
                ('Route 101 - Central Station', '08:00 AM', 40, 38, 0, 'Robert D.', 'Michael S.'),
                ('Route 205 - Business Park', '09:15 AM', 50, 50, 12, 'Sarah C.', 'James T.'),
                ('Route 304 - City Hospital', '10:30 AM', 30, 15, 0, 'David W.', 'Linda K.')
            ]
            c.executemany("INSERT INTO schedules (route_name, departure_time, capacity, booked_seats, overflow_count, driver_name, conductor_name) VALUES (?, ?, ?, ?, ?, ?, ?)", schedules)
            conn.commit()

init_db()

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        db = get_db()
        username = request.form.get('username')
        password = request.form.get('password')
        user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['name'] = user['name']
            session['role'] = user['role']
            if user['role'] == 'staff':
                return redirect('/staff/dashboard')
            else:
                return redirect('/passenger/dashboard')
        flash('Invalid username or password. Please try again.', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        db = get_db()
        username = request.form.get('username')
        password = request.form.get('password')
        name = request.form.get('name')
        role = request.form.get('role')
        try:
            db.execute("INSERT INTO users (username, password, name, role) VALUES (?, ?, ?, ?)", 
                       (username, generate_password_hash(password), name, role))
            db.commit()
            flash('Registration successful! Please login to continue.', 'success')
            return redirect('/login')
        except sqlite3.IntegrityError:
            flash('This username is already taken. Please choose another.', 'error')
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

# --- PASSENGER ROUTES ---
@app.route('/passenger/dashboard')
@login_required
@role_required('passenger')
def passenger_dashboard():
    db = get_db()
    schedules = db.execute("SELECT * FROM schedules").fetchall()
    return render_template('passenger/dashboard.html', schedules=schedules)

@app.route('/passenger/book/<int:schedule_id>', methods=['POST'])
@login_required
@role_required('passenger')
def book_ticket(schedule_id):
    db = get_db()
    schedule = db.execute("SELECT * FROM schedules WHERE id = ?", (schedule_id,)).fetchone()
    
    if schedule['booked_seats'] < schedule['capacity']:
        db.execute("UPDATE schedules SET booked_seats = booked_seats + 1 WHERE id = ?", (schedule_id,))
        db.execute("INSERT INTO bookings (user_id, schedule_id, status) VALUES (?, ?, 'confirmed')", (session['user_id'], schedule_id))
        db.commit()
        flash('Seat confirmed! Your ticket has been booked successfully.', 'success')
    else:
        db.execute("UPDATE schedules SET overflow_count = overflow_count + 1 WHERE id = ?", (schedule_id,))
        db.execute("INSERT INTO bookings (user_id, schedule_id, status) VALUES (?, ?, 'overflow')", (session['user_id'], schedule_id))
        db.commit()
        flash('Notice: This bus is full. You have been added to the waitlist. This helps our AI schedule more buses in the future.', 'warning')
        
    return redirect('/passenger/dashboard')

# --- STAFF ROUTES ---
@app.route('/staff/dashboard')
@login_required
@role_required('staff')
def staff_dashboard():
    db = get_db()
    schedules = db.execute("SELECT * FROM schedules").fetchall()
    return render_template('staff/dashboard.html', schedules=schedules)

@app.route('/staff/edit', methods=['GET', 'POST'])
@login_required
@role_required('staff')
def staff_edit():
    db = get_db()
    if request.method == 'POST':
        route_name = request.form.get('route_name')
        departure_time = request.form.get('departure_time')
        capacity = request.form.get('capacity')
        driver_name = request.form.get('driver_name')
        conductor_name = request.form.get('conductor_name')
        db.execute("INSERT INTO schedules (route_name, departure_time, capacity, driver_name, conductor_name) VALUES (?, ?, ?, ?, ?)",
                   (route_name, departure_time, capacity, driver_name, conductor_name))
        db.commit()
        flash('New schedule added successfully to the system.', 'success')
        return redirect('/staff/dashboard')
    
    schedules = db.execute("SELECT * FROM schedules").fetchall()
    return render_template('staff/edit.html', schedules=schedules)

if __name__ == '__main__':
    app.run(debug=True)
