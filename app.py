
from flask import Flask, render_template, request, redirect, session, flash, g, send_file, jsonify
import sqlite3
import os
import io
import json
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors

app = Flask(__name__)
app.secret_key = 'super_secret_transit_v7'
DATABASE = 'busbalance_v7.db'

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
                flash("Unauthorized access. Staff only.", "error")
                return redirect("/")
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def passenger_access(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get("role") not in ['passenger', 'staff']:
            flash("Unauthorized access.", "error")
            return redirect("/")
        return f(*args, **kwargs)
    return decorated_function

def init_db():
    if not os.path.exists(DATABASE):
        with sqlite3.connect(DATABASE) as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT, name TEXT, role TEXT)''')
            c.execute('''CREATE TABLE schedules (id INTEGER PRIMARY KEY, route_name TEXT, travel_date TEXT, departure_time TEXT, capacity INTEGER, booked_seats INTEGER DEFAULT 0, overflow_count INTEGER DEFAULT 0, driver_name TEXT, conductor_name TEXT)''')
            c.execute('''CREATE TABLE bookings (id INTEGER PRIMARY KEY, user_id INTEGER, schedule_id INTEGER, status TEXT, seat_number TEXT, total_cost REAL, booking_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
            
            c.execute("INSERT INTO users (username, password, name, role) VALUES (?, ?, ?, ?)", ('admin', generate_password_hash('admin'), 'Transport Manager', 'staff'))
            c.execute("INSERT INTO users (username, password, name, role) VALUES (?, ?, ?, ?)", ('user', generate_password_hash('user'), 'John Passenger', 'passenger'))
            
            # Generate 7 days of dummy data
            schedules = []
            routes = [
                ('Route 101 - Central Hub', '08:00 AM', 40, 'Robert D.', 'Michael S.'),
                ('Route 205 - Financial Dist', '09:15 AM', 40, 'Sarah C.', 'James T.'),
                ('Route 304 - Suburbs', '10:30 AM', 32, 'David W.', 'Linda K.')
            ]
            
            for i in range(7):
                date_str = (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
                for r in routes:
                    schedules.append((r[0], date_str, r[1], r[2], 0, 0, r[3], r[4]))
                    
            c.executemany("INSERT INTO schedules (route_name, travel_date, departure_time, capacity, booked_seats, overflow_count, driver_name, conductor_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", schedules)
            conn.commit()

init_db()

def get_minutes_to_departure(travel_date_str, time_str):
    """Calculates minutes remaining until departure. Returns 9999 if parsing fails."""
    dt_str = f"{travel_date_str} {time_str}"
    try:
        departure_dt = datetime.strptime(dt_str, "%Y-%m-%d %I:%M %p")
        now = datetime.now()
        return (departure_dt - now).total_seconds() / 60.0
    except ValueError:
        return 9999

@app.route('/')
@app.route('/home')
def home():
    db = get_db()
    today_str = datetime.now().strftime('%Y-%m-%d')
    schedules = db.execute("SELECT * FROM schedules WHERE travel_date = ? ORDER BY departure_time", (today_str,)).fetchall()
    
    today_schedules = []
    for s in schedules:
        mins = get_minutes_to_departure(s['travel_date'], s['departure_time'])
        if mins > 0: # Only pass buses that have not departed yet
            s_dict = dict(s)
            s_dict['minutes_to_departure'] = mins
            today_schedules.append(s_dict)
            
    return render_template('index.html', today_schedules=today_schedules, today_date=today_str)
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
            return redirect('/passenger/search')
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

# Autocomplete API route
@app.route('/api/routes')
@login_required
@passenger_access
def api_routes():
    query = request.args.get('q', '').lower()
    db = get_db()
    if query:
        routes = db.execute("SELECT DISTINCT route_name FROM schedules WHERE LOWER(route_name) LIKE ?", (f"%{query}%",)).fetchall()
    else:
        routes = db.execute("SELECT DISTINCT route_name FROM schedules").fetchall()
    return jsonify([r['route_name'] for r in routes])

@app.route('/passenger/search', methods=['GET', 'POST'])
@login_required
@passenger_access
def search():
    if request.method == 'POST':
        route_query = request.form.get('route_query', '').lower()
        travel_date = request.form.get('travel_date')
        try:
            passenger_count = int(request.form.get('passenger_count'))
        except:
            passenger_count = 1
        db = get_db()
        query = "SELECT * FROM schedules WHERE travel_date = ?"
        params = [travel_date]
        if route_query:
            query += " AND LOWER(route_name) LIKE ?"
            params.append(f"%{route_query}%")
        # ... inside search() POST block ...
        raw_schedules = db.execute(query, params).fetchall()
        
        schedules = []
        for s in raw_schedules:
            mins = get_minutes_to_departure(s['travel_date'], s['departure_time'])
            if mins > 0: # Only pass future buses
                s_dict = dict(s)
                s_dict['minutes_to_departure'] = mins
                schedules.append(s_dict)
                
        return render_template('passenger/dashboard.html', schedules=schedules, passenger_count=passenger_count, travel_date=travel_date)
        
    # Default to tomorrow for the calendar
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    max_date = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
    return render_template('passenger/search.html', min_date=tomorrow, max_date=max_date)

@app.route('/passenger/select_seat/<int:schedule_id>')
@login_required
@passenger_access
def select_seat(schedule_id):
    db = get_db()
    passenger_count = request.args.get('count', 1)
    schedule = db.execute("SELECT * FROM schedules WHERE id = ?", (schedule_id,)).fetchone()
    
    # 15-Minute Booking Lock
    mins = get_minutes_to_departure(schedule['travel_date'], schedule['departure_time'])
    if mins <= 15:
        flash("Booking is closed for this route (departs in less than 15 minutes).", "error")
        return redirect('/passenger/search')
    
    if schedule['booked_seats'] >= schedule['capacity']:
        return render_template('passenger/bus_full.html', schedule=schedule, mins=mins)
        
    # ... rest of the original function ...
    bookings = db.execute("SELECT seat_number FROM bookings WHERE schedule_id = ? AND status IN ('confirmed', 'pending')", (schedule_id,)).fetchall()
    booked_seats = set()
    for b in bookings:
        if b['seat_number']:
            booked_seats.update(json.loads(b['seat_number']))
    
    return render_template('passenger/select_seat.html', schedule=schedule, booked_seats=list(booked_seats), passenger_count=passenger_count)

@app.route('/passenger/process_waitlist/<int:schedule_id>', methods=['POST'])
@login_required
@passenger_access
def process_waitlist(schedule_id):
    db = get_db()
    schedule = db.execute("SELECT * FROM schedules WHERE id = ?", (schedule_id,)).fetchone()
    
    # 2-Hour Waitlist Lock
    mins = get_minutes_to_departure(schedule['travel_date'], schedule['departure_time'])
    if mins <= 120:
        flash('Waitlist is closed. We require 2 hours notice to deploy additional buses.', 'error')
        return redirect('/passenger/search')
        
    db.execute("UPDATE schedules SET overflow_count = overflow_count + 1 WHERE id = ?", (schedule_id,))
    # ... rest of original function ...
    db.execute("INSERT INTO bookings (user_id, schedule_id, status, seat_number, total_cost) VALUES (?, ?, 'overflow', '', 0)", (session['user_id'], schedule_id))
    db.commit()
    flash('Added to waitlist. This helps our AI improve schedules.', 'warning')
    return redirect('/passenger/history')

@app.route('/passenger/checkout', methods=['POST'])
@login_required
@passenger_access
def checkout():
    db = get_db()
    schedule_id = request.form.get('schedule_id')
    passenger_count = int(request.form.get('passenger_count', 1))
    seat_numbers = request.form.getlist('seat_number') 
    
    if len(seat_numbers) != passenger_count:
        flash(f"You must select exactly {passenger_count} seat(s).", "error")
        return redirect(f'/passenger/select_seat/{schedule_id}?count={passenger_count}')
        
    requested_set = set([int(s) for s in seat_numbers])
    
    existing = db.execute("SELECT seat_number FROM bookings WHERE schedule_id = ? AND status IN ('confirmed', 'pending')", (schedule_id,)).fetchall()
    booked_set = set()
    for b in existing:
        if b['seat_number']:
            booked_set.update(json.loads(b['seat_number']))
            
    if requested_set.intersection(booked_set):
        flash("Alert: One or more selected seats were just booked. Please choose new seats.", "error")
        return redirect(f'/passenger/select_seat/{schedule_id}?count={passenger_count}')
    
    seat_str = json.dumps(list(requested_set))
    total_cost = len(requested_set) * 2.50
    
    cursor = db.cursor()
    cursor.execute("INSERT INTO bookings (user_id, schedule_id, status, seat_number, total_cost) VALUES (?, ?, 'pending', ?, ?)", 
                   (session['user_id'], schedule_id, seat_str, total_cost))
    booking_id = cursor.lastrowid
    db.commit()
    
    return redirect(f'/passenger/payment/{booking_id}')

@app.route('/passenger/payment/<int:booking_id>', methods=['GET', 'POST'])
@login_required
@passenger_access
def payment(booking_id):
    db = get_db()
    if request.method == 'POST':
        booking = db.execute("SELECT * FROM bookings WHERE id = ?", (booking_id,)).fetchone()
        num_seats = len(json.loads(booking['seat_number'])) if booking['seat_number'] else 0
        
        db.execute("UPDATE schedules SET booked_seats = booked_seats + ? WHERE id = ?", (num_seats, booking['schedule_id']))
        db.execute("UPDATE bookings SET status = 'confirmed' WHERE id = ?", (booking_id,))
        db.commit()
        
        flash('Payment Successful! Ticket confirmed.', 'success')
        return redirect('/passenger/history')
        
    booking_details = db.execute('''SELECT b.*, s.route_name, s.departure_time, s.travel_date 
                                    FROM bookings b JOIN schedules s ON b.schedule_id = s.id 
                                    WHERE b.id = ?''', (booking_id,)).fetchone()
    num_seats = len(json.loads(booking_details['seat_number'])) if booking_details['seat_number'] else 0
    formatted_seats = ", ".join(map(str, json.loads(booking_details['seat_number']))) if booking_details['seat_number'] else ''
    
    return render_template('passenger/payment.html', booking=booking_details, num_seats=num_seats, formatted_seats=formatted_seats)

@app.route('/passenger/history')
@login_required
@passenger_access
def passenger_history():
    db = get_db()
    history = db.execute('''
        SELECT b.id, b.status, b.seat_number, b.total_cost, b.booking_time, s.route_name, s.departure_time, s.travel_date 
        FROM bookings b JOIN schedules s ON b.schedule_id = s.id 
        WHERE b.user_id = ? ORDER BY b.booking_time DESC
    ''', (session['user_id'],)).fetchall()
    
    formatted_history = []
    for h in history:
        h_dict = dict(h)
        h_dict['seats'] = ", ".join(map(str, json.loads(h['seat_number']))) if h['seat_number'] else '--'
        formatted_history.append(h_dict)
        
    return render_template('passenger/history.html', history=formatted_history)

@app.route('/passenger/download_ticket/<int:booking_id>')
@login_required
@passenger_access
def download_ticket(booking_id):
    db = get_db()
    ticket = db.execute('''
        SELECT b.id, b.status, b.seat_number, b.total_cost, b.booking_time, s.route_name, s.departure_time, s.travel_date, u.name 
        FROM bookings b 
        JOIN schedules s ON b.schedule_id = s.id 
        JOIN users u ON b.user_id = u.id
        WHERE b.id = ? AND b.user_id = ? AND b.status = 'confirmed'
    ''', (booking_id, session['user_id'])).fetchone()
    
    if not ticket:
        flash("Ticket not found or not confirmed.", "error")
        return redirect('/passenger/history')

    buffer = io.BytesIO()
    
    # --- CUSTOM PAGE SIZE (Perfectly wraps the ticket for Mobile/Digital viewing) ---
    page_width = 580
    page_height = 220
    p = canvas.Canvas(buffer, pagesize=(page_width, page_height))
    
    # Dimensions & Positioning (Shifted origin down to remove white space)
    ticket_width = 540
    ticket_height = 200
    x_offset = 20
    y_offset = 10
    
    # Outer Border
    p.setFillColor(colors.HexColor("#e2e8f0"))
    p.roundRect(x_offset + 2, y_offset - 2, ticket_width, ticket_height, 12, fill=1, stroke=0)
    
    # Main Ticket Background
    p.setFillColor(colors.white)
    p.setStrokeColor(colors.HexColor("#cbd5e1"))
    p.setLineWidth(1)
    p.roundRect(x_offset, y_offset, ticket_width, ticket_height, 12, fill=1, stroke=1)
    
    # Header Banner
    p.setFillColor(colors.HexColor("#0f172a"))
    p.roundRect(x_offset, y_offset + ticket_height - 40, ticket_width, 40, 12, fill=1, stroke=0)
    p.rect(x_offset, y_offset + ticket_height - 40, ticket_width, 20, fill=1, stroke=0)
    
    # Header Text
    p.setFillColor(colors.white)
    p.setFont("Helvetica-Bold", 16)
    p.drawString(x_offset + 20, y_offset + ticket_height - 26, "BusBalance Municipal Transit")
    
    p.setFont("Helvetica-Bold", 12)
    p.setFillColor(colors.HexColor("#38bdf8"))
    p.drawRightString(x_offset + ticket_width - 20, y_offset + ticket_height - 25, "OFFICIAL E-TICKET")
    
    # Perforated Divider Line
    p.setStrokeColor(colors.HexColor("#94a3b8"))
    p.setDash(4, 4)
    p.line(x_offset + 390, y_offset, x_offset + 390, y_offset + ticket_height - 40)
    p.setDash(1, 0)
    
    # --- LEFT SECTION: PROFESSIONAL TABLE/GRID ---
    grid_x = x_offset + 20
    grid_y = y_offset + 20
    grid_w = 350
    row_h = 35
    
    # Draw Grid Lines
    p.setStrokeColor(colors.HexColor("#cbd5e1"))
    p.rect(grid_x, grid_y, grid_w, row_h * 3) # Outer box
    p.line(grid_x, grid_y + row_h, grid_x + grid_w, grid_y + row_h) # Horizontal 1
    p.line(grid_x, grid_y + row_h * 2, grid_x + grid_w, grid_y + row_h * 2) # Horizontal 2
    p.line(grid_x + 175, grid_y + row_h, grid_x + 175, grid_y + row_h * 3) # Vertical split for top 2 rows
    p.line(grid_x + 116, grid_y, grid_x + 116, grid_y + row_h) # Vertical split 1 for bottom row
    p.line(grid_x + 233, grid_y, grid_x + 233, grid_y + row_h) # Vertical split 2 for bottom row
    
    # Populate Grid Headers (Light Gray Text)
    p.setFillColor(colors.HexColor("#64748b"))
    p.setFont("Helvetica-Bold", 8)
    p.drawString(grid_x + 10, grid_y + (row_h * 3) - 15, "PASSENGER NAME")
    p.drawString(grid_x + 185, grid_y + (row_h * 3) - 15, "BOOKING REFERENCE")
    
    p.drawString(grid_x + 10, grid_y + (row_h * 2) - 15, "ROUTE DESIGNATION")
    p.drawString(grid_x + 185, grid_y + (row_h * 2) - 15, "DEPARTURE DATE")
    
    p.drawString(grid_x + 10, grid_y + row_h - 15, "DEPARTURE TIME")
    p.drawString(grid_x + 126, grid_y + row_h - 15, "TOTAL PAID")
    p.drawString(grid_x + 243, grid_y + row_h - 15, "ISSUED ON")
    
    # Populate Grid Data (Dark Navy Text)
    p.setFillColor(colors.HexColor("#0f172a"))
    
    # Row 3 (Top) Data
    p.setFont("Helvetica-Bold", 12)
    p.drawString(grid_x + 10, grid_y + (row_h * 3) - 30, str(ticket['name']))
    p.drawString(grid_x + 185, grid_y + (row_h * 3) - 30, f"#{ticket['id']}")
    
    # Row 2 (Middle) Data
    route = str(ticket['route_name'])
    if len(route) > 24: route = route[:21] + "..."
    p.drawString(grid_x + 10, grid_y + (row_h * 2) - 30, route)
    p.drawString(grid_x + 185, grid_y + (row_h * 2) - 30, str(ticket['travel_date']))
    
    # Row 1 (Bottom) Data
    p.setFont("Helvetica-Bold", 11)
    p.drawString(grid_x + 10, grid_y + row_h - 30, str(ticket['departure_time']))
    p.drawString(grid_x + 126, grid_y + row_h - 30, f"${ticket['total_cost']:.2f}")
    p.drawString(grid_x + 243, grid_y + row_h - 30, str(ticket['booking_time'][:10]))
    
    # --- RIGHT SECTION: STUB & BARCODE ---
    p.setFillColor(colors.HexColor("#64748b"))
    p.setFont("Helvetica-Bold", 9)
    p.drawCentredString(x_offset + 465, y_offset + 130, "VALID FOR SEAT(S)")
    
    # Emphasized Seat Box
    p.setFillColor(colors.HexColor("#e2e8f0"))
    p.roundRect(x_offset + 410, y_offset + 95, 110, 25, 4, fill=1, stroke=0)
    p.setFillColor(colors.HexColor("#e11d48"))
    
    seats = ", ".join(map(str, json.loads(ticket['seat_number']))) if ticket['seat_number'] else 'WL'
    if len(seats) > 12: p.setFont("Helvetica-Bold", 10)
    else: p.setFont("Helvetica-Bold", 14)
    p.drawCentredString(x_offset + 465, y_offset + 103, seats)
    
    # Procedural Mock Barcode
    p.setFillColor(colors.black)
    import random
    random.seed(ticket['id'])
    for i in range(25):
        w = random.randint(1, 4)
        p.rect(x_offset + 410 + (i*4.4), y_offset + 40, w, 35, fill=1, stroke=0)
        
    p.setFillColor(colors.HexColor("#94a3b8"))
    p.setFont("Courier", 8)
    p.drawCentredString(x_offset + 465, y_offset + 25, f"AUTH-{ticket['id']}{ticket['booking_time'][5:7]}")
    
    p.showPage()
    p.save()
    buffer.seek(0)
    
    return send_file(buffer, as_attachment=True, download_name=f"BusBalance_Ticket_{booking_id}.pdf", mimetype='application/pdf')

@app.route('/staff/dashboard')
@login_required
@role_required('staff')
def staff_dashboard():
    db = get_db()
    schedules = db.execute("SELECT * FROM schedules ORDER BY travel_date, departure_time").fetchall()
    return render_template('staff/dashboard.html', schedules=schedules)

@app.route('/staff/add', methods=['GET', 'POST'])
@login_required
@role_required('staff')
def staff_add():
    if request.method == 'POST':
        db = get_db()
        route_name = request.form.get('route_name')
        travel_date = request.form.get('travel_date')
        departure_time = request.form.get('departure_time')
        capacity = request.form.get('capacity')
        driver_name = request.form.get('driver_name')
        conductor_name = request.form.get('conductor_name')
        db.execute("INSERT INTO schedules (route_name, travel_date, departure_time, capacity, driver_name, conductor_name) VALUES (?, ?, ?, ?, ?, ?)",
                   (route_name, travel_date, departure_time, capacity, driver_name, conductor_name))
        db.commit()
        flash('Schedule added.', 'success')
        return redirect('/staff/dashboard')
    
    # Default to tomorrow for new schedules
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    return render_template('staff/form.html', action="Add", schedule=None, default_date=tomorrow)

@app.route('/staff/edit/<int:schedule_id>', methods=['GET', 'POST'])
@login_required
@role_required('staff')
def staff_edit(schedule_id):
    db = get_db()
    if request.method == 'POST':
        route_name = request.form.get('route_name')
        travel_date = request.form.get('travel_date')
        departure_time = request.form.get('departure_time')
        capacity = request.form.get('capacity')
        driver_name = request.form.get('driver_name')
        conductor_name = request.form.get('conductor_name')
        db.execute('''UPDATE schedules SET route_name=?, travel_date=?, departure_time=?, capacity=?, driver_name=?, conductor_name=? WHERE id=?''',
                   (route_name, travel_date, departure_time, capacity, driver_name, conductor_name, schedule_id))
        db.commit()
        flash('Schedule updated successfully.', 'success')
        return redirect('/staff/dashboard')
        
    schedule = db.execute("SELECT * FROM schedules WHERE id = ?", (schedule_id,)).fetchone()
    return render_template('staff/form.html', action="Edit", schedule=schedule, default_date=None)

