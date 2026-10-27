
from flask import Flask, render_template, request, redirect, session, flash, g, send_file
import sqlite3
import os
import io
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

# PDF Generation imports
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors

app = Flask(__name__)
app.secret_key = 'super_secret_transit_key_v3'
DATABASE = 'busbalance_v3.db'

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
                flash("Unauthorized access.", "error")
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
            # Added seat_number and timestamp for history
            c.execute('''CREATE TABLE bookings (id INTEGER PRIMARY KEY, user_id INTEGER, schedule_id INTEGER, status TEXT, seat_number INTEGER, booking_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
            
            # Dummy data
            c.execute("INSERT INTO users (username, password, name, role) VALUES (?, ?, ?, ?)", ('admin', generate_password_hash('admin'), 'Transport Manager', 'staff'))
            c.execute("INSERT INTO users (username, password, name, role) VALUES (?, ?, ?, ?)", ('user', generate_password_hash('user'), 'John Passenger', 'passenger'))
            
            schedules = [
                ('Route 101 - Central Hub', '08:00 AM', 40, 0, 0, 'Robert D.', 'Michael S.'),
                ('Route 205 - Financial Dist', '09:15 AM', 40, 0, 0, 'Sarah C.', 'James T.'),
                ('Route 304 - Suburbs', '10:30 AM', 32, 0, 0, 'David W.', 'Linda K.')
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
        flash('Invalid credentials. Try again.', 'error')
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
            flash('Registration successful!', 'success')
            return redirect('/login')
        except sqlite3.IntegrityError:
            flash('Username taken.', 'error')
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

@app.route('/passenger/select_seat/<int:schedule_id>')
@login_required
@role_required('passenger')
def select_seat(schedule_id):
    db = get_db()
    schedule = db.execute("SELECT * FROM schedules WHERE id = ?", (schedule_id,)).fetchone()
    
    # Check if full
    if schedule['booked_seats'] >= schedule['capacity']:
        return render_template('passenger/bus_full.html', schedule=schedule)

    # Get already booked seats for this schedule
    bookings = db.execute("SELECT seat_number FROM bookings WHERE schedule_id = ? AND status IN ('confirmed', 'pending')", (schedule_id,)).fetchall()
    booked_seats = [b['seat_number'] for b in bookings if b['seat_number'] is not None]
    
    return render_template('passenger/select_seat.html', schedule=schedule, booked_seats=booked_seats)

@app.route('/passenger/process_waitlist/<int:schedule_id>', methods=['POST'])
@login_required
@role_required('passenger')
def process_waitlist(schedule_id):
    db = get_db()
    db.execute("UPDATE schedules SET overflow_count = overflow_count + 1 WHERE id = ?", (schedule_id,))
    db.execute("INSERT INTO bookings (user_id, schedule_id, status) VALUES (?, ?, 'overflow')", (session['user_id'], schedule_id))
    db.commit()
    flash('Added to waitlist. This data helps our AI improve tomorrow\'s schedules.', 'warning')
    return redirect('/passenger/history')

@app.route('/passenger/checkout', methods=['POST'])
@login_required
@role_required('passenger')
def checkout():
    db = get_db()
    schedule_id = request.form.get('schedule_id')
    seat_number = request.form.get('seat_number')
    
    if not seat_number:
        flash("You must select a seat.", "error")
        return redirect(f'/passenger/select_seat/{schedule_id}')
        
    # Create a pending booking
    cursor = db.cursor()
    cursor.execute("INSERT INTO bookings (user_id, schedule_id, status, seat_number) VALUES (?, ?, 'pending', ?)", 
                   (session['user_id'], schedule_id, seat_number))
    booking_id = cursor.lastrowid
    db.commit()
    
    return redirect(f'/passenger/payment/{booking_id}')

@app.route('/passenger/payment/<int:booking_id>', methods=['GET', 'POST'])
@login_required
@role_required('passenger')
def payment(booking_id):
    db = get_db()
    if request.method == 'POST':
        # Process mock payment
        booking = db.execute("SELECT * FROM bookings WHERE id = ?", (booking_id,)).fetchone()
        
        # Update schedule capacity and confirm booking
        db.execute("UPDATE schedules SET booked_seats = booked_seats + 1 WHERE id = ?", (booking['schedule_id'],))
        db.execute("UPDATE bookings SET status = 'confirmed' WHERE id = ?", (booking_id,))
        db.commit()
        
        flash('Payment Successful! Ticket confirmed.', 'success')
        return redirect('/passenger/history')
        
    # Render mock payment page
    booking_details = db.execute('''SELECT b.*, s.route_name, s.departure_time 
                                    FROM bookings b JOIN schedules s ON b.schedule_id = s.id 
                                    WHERE b.id = ?''', (booking_id,)).fetchone()
    return render_template('passenger/payment.html', booking=booking_details)

@app.route('/passenger/history')
@login_required
@role_required('passenger')
def passenger_history():
    db = get_db()
    history = db.execute('''
        SELECT b.id, b.status, b.seat_number, b.booking_time, s.route_name, s.departure_time, s.driver_name 
        FROM bookings b 
        JOIN schedules s ON b.schedule_id = s.id 
        WHERE b.user_id = ? 
        ORDER BY b.booking_time DESC
    ''', (session['user_id'],)).fetchall()
    return render_template('passenger/history.html', history=history)

@app.route('/passenger/download_ticket/<int:booking_id>')
@login_required
@role_required('passenger')
def download_ticket(booking_id):
    db = get_db()
    ticket = db.execute('''
        SELECT b.id, b.status, b.seat_number, b.booking_time, s.route_name, s.departure_time, u.name 
        FROM bookings b 
        JOIN schedules s ON b.schedule_id = s.id 
        JOIN users u ON b.user_id = u.id
        WHERE b.id = ? AND b.user_id = ? AND b.status = 'confirmed'
    ''', (booking_id, session['user_id'])).fetchone()
    
    if not ticket:
        flash("Ticket not found or not confirmed.", "error")
        return redirect('/passenger/history')

    # Generate PDF using ReportLab
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    
    # Corporate Header
    p.setFillColor(colors.HexColor("#003366"))
    p.rect(0, 730, 612, 62, fill=1)
    p.setFillColor(colors.white)
    p.setFont("Helvetica-Bold", 24)
    p.drawString(40, 750, "BusBalance Transit Authority")
    p.setFont("Helvetica", 12)
    p.drawString(40, 735, "Official E-Ticket")
    
    # Body
    p.setFillColor(colors.black)
    p.setFont("Helvetica-Bold", 16)
    p.drawString(40, 680, "Passenger Details")
    p.line(40, 675, 570, 675)
    
    p.setFont("Helvetica", 12)
    p.drawString(40, 650, f"Passenger Name: {ticket['name']}")
    p.drawString(40, 630, f"Booking ID: #{ticket['id']}")
    p.drawString(40, 610, f"Date Issued: {ticket['booking_time']}")
    
    p.setFont("Helvetica-Bold", 16)
    p.drawString(40, 560, "Journey Details")
    p.line(40, 555, 570, 555)
    
    p.setFont("Helvetica", 14)
    p.drawString(40, 530, f"Route: {ticket['route_name']}")
    p.drawString(40, 505, f"Departure: {ticket['departure_time']}")
    
    p.setFont("Helvetica-Bold", 16)
    p.setFillColor(colors.HexColor("#28a745"))
    p.drawString(40, 470, f"Seat Number: {ticket['seat_number']}")
    
    p.setFillColor(colors.black)
    p.setFont("Helvetica-Oblique", 10)
    p.drawString(40, 100, "Please present this digital ticket upon boarding.")
    p.drawString(40, 85, "Generated securely by BusBalance AI System.")
    
    p.showPage()
    p.save()
    buffer.seek(0)
    
    return send_file(buffer, as_attachment=True, download_name=f"BusBalance_Ticket_{booking_id}.pdf", mimetype='application/pdf')

# --- STAFF ROUTES ---
@app.route('/staff/dashboard')
@login_required
@role_required('staff')
def staff_dashboard():
    db = get_db()
    schedules = db.execute("SELECT * FROM schedules").fetchall()
    return render_template('staff/dashboard.html', schedules=schedules)

@app.route('/staff/add', methods=['GET', 'POST'])
@login_required
@role_required('staff')
def staff_add():
    if request.method == 'POST':
        db = get_db()
        route_name = request.form.get('route_name')
        departure_time = request.form.get('departure_time')
        capacity = request.form.get('capacity')
        driver_name = request.form.get('driver_name')
        conductor_name = request.form.get('conductor_name')
        db.execute("INSERT INTO schedules (route_name, departure_time, capacity, driver_name, conductor_name) VALUES (?, ?, ?, ?, ?)",
                   (route_name, departure_time, capacity, driver_name, conductor_name))
        db.commit()
        flash('Schedule added.', 'success')
        return redirect('/staff/dashboard')
    return render_template('staff/form.html', action="Add", schedule=None)

@app.route('/staff/edit/<int:schedule_id>', methods=['GET', 'POST'])
@login_required
@role_required('staff')
def staff_edit(schedule_id):
    db = get_db()
    if request.method == 'POST':
        route_name = request.form.get('route_name')
        departure_time = request.form.get('departure_time')
        capacity = request.form.get('capacity')
        driver_name = request.form.get('driver_name')
        conductor_name = request.form.get('conductor_name')
        db.execute('''UPDATE schedules SET route_name=?, departure_time=?, capacity=?, driver_name=?, conductor_name=? WHERE id=?''',
                   (route_name, departure_time, capacity, driver_name, conductor_name, schedule_id))
        db.commit()
        flash('Schedule updated successfully.', 'success')
        return redirect('/staff/dashboard')
        
    schedule = db.execute("SELECT * FROM schedules WHERE id = ?", (schedule_id,)).fetchone()
    return render_template('staff/form.html', action="Edit", schedule=schedule)

if __name__ == '__main__':
    app.run(debug=True)
