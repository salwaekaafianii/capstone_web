from flask import (
    Flask, render_template, redirect, url_for,jsonify,
    request, session, flash, get_flashed_messages,
)
import mysql.connector 

app = Flask(__name__)
app.config['SECRET_KEY'] = 'kunci-rahasia-teman-tukang-yang-kuat'

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="capst_web"
)
cursor = db.cursor(dictionary=True)

@app.route('/')
def index():
    return redirect(url_for('dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email    = request.form.get('email')
        password = request.form.get('password')

        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()

        # Email tidak ditemukan
        if not user:
            flash("Email tidak terdaftar!", "danger")
            return redirect(url_for('login'))

        # Password salah
        if user['password'] != password:
            flash("Password salah!", "danger")
            return redirect(url_for('login'))

        # Jika login benar → simpan session
        session['user_id']    = user['id_users']
        session['user_email'] = user['email']
        session['user_role']  = user['role']

        flash("Login berhasil!", "success")

        # Redirect berdasarkan role
        if user['role'] == "admin":
            return redirect(url_for('admin_dashboard'))
        elif user['role'] == "tukang":
            return redirect(url_for('tukang_dashboard'))
        else:
            return redirect(url_for('dashboard'))

    return render_template('login.html')

@app.route('/admin')
def admin_dashboard():
    if 'user_role' not in session or session['user_role'] != 'admin':
        flash("Akses ditolak!", "danger")
        return redirect(url_for('login'))

    cursor.execute("SELECT COUNT(*) AS total FROM users WHERE role='tukang'")
    total_tukang = cursor.fetchone()['total']

    cursor.execute("SELECT COUNT(*) AS total FROM users WHERE role='customer'")
    total_customer = cursor.fetchone()['total']

    return render_template('admin/admin_dashboard.html',
                           total_tukang=total_tukang,
                           total_customer=total_customer)


@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email    = request.form['email']
        password = request.form['password']

        if len(password) < 6 or len(password) > 8:
            flash("Password harus 6-8 karakter!", "danger")
            return redirect(url_for('register'))
        
        cur = db.cursor()
        cur.execute(
            "INSERT INTO users (username, email, password, role) VALUES (%s, %s, %s, 'customer')",
            (username, email, password)
        )
        db.commit()

        flash("Registrasi berhasil! Silakan login.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/riwayat-pesanan')
def riwayat_pesanan():
    if 'user_id' not in session:  
        flash("Anda harus login untuk melihat riwayat pesanan.", "warning")
        return redirect(url_for('login'))
    
    simulated_orders = {
        101: {'layanan': 'Perbaikan Pipa Bocor', 'tukang': 'Ahmad Imam', 'tanggal': '15/10/2025', 'status': 'Menunggu'},
        102: {'layanan': 'Pemasangan Keramik', 'tukang': 'Ibu Rina', 'tanggal': '01/11/2025', 'status': 'Selesai'},
    }

    return render_template(
        'riwayat_pesanan.html',
        orders=simulated_orders,
        active_page='riwayat_pesanan'
    )

@app.route('/ulasan/<int:order_id>', methods=['GET', 'POST'])
def tulis_ulasan(order_id):
    if request.method == 'POST':
        return redirect(url_for('riwayat_pesanan'))
    
    return render_template('tulis_ulasan.html', order_id=order_id)

@app.route('/deteksi', methods=['GET', 'POST'])
def deteksi():
    get_flashed_messages()

    if 'user_id' not in session:
        flash("Anda harus login untuk menggunakan fitur deteksi.")
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        file = request.files.get('file')
        
        if not file or file.filename == '':
            flash('Tidak ada file yang dipilih.')
            return redirect(request.url)

        flash('File berhasil diupload.')
        return redirect(url_for('deteksi_hasil'))
    
    return render_template(
        'deteksi.html',
        active_page='deteksi'
    )

@app.route('/deteksi-hasil')
def deteksi_hasil():
    hasil = {
        'gambar_rusak': 'https://placehold.co/600x400/808080/FFFFFF?text=Dinding+Retak',
        'analisis': (
            'Ditemukan kerusakan struktural berupa retakan lebar, kemungkinan akibat '
            'pergerakan tanah atau beban berlebih. Perlu evaluasi fondasi.'
        ),
        'faktor': (
            'Terjadi karena fondasi penurunan tidak merata dan kelembaban tinggi di area retak.'
        )
    }
    
    return render_template(
        'deteksi_hasil.html',
        hasil=hasil,
        active_page='deteksi'
    )

@app.route("/rekomendasi")
def rekomendasi():
    simulated_tukang = [
        {"id": 1, "nama": "Budi Santoso", "keahlian": "Pengecatan & Plafon",
         "rating": 4.8, "jumlah_review": 12, "pengalaman": "10 tahun pengalaman",
         "foto": "https://placehold.co/150x150"},
        {"id": 2, "nama": "Siti Aminah", "keahlian": "Listrik & Instalasi",
         "rating": 4.5, "jumlah_review": 8, "pengalaman": "7 tahun pengalaman",
         "foto": "https://placehold.co/150x150"},
        {"id": 3, "nama": "Agus Wijaya", "keahlian": "Pemasangan Keramik",
         "rating": 4.7, "jumlah_review": 10, "pengalaman": "5 tahun pengalaman",
         "foto": "https://placehold.co/150x150"},
    ]
    
    return render_template("rekomendasi.html", tukangs=simulated_tukang)

@app.route("/lihat-tukang/<int:tukang_id>")
def lihat_tukang(tukang_id):
    simulated_tukang = [
        {"id": 1, "nama": "Budi Santoso", "keahlian": "Pengecatan & Plafon",
         "rating": 4.8, "jumlah_review": 12, "pengalaman": "10 tahun pengalaman",
         "foto": "https://placehold.co/150x150"},
        {"id": 2, "nama": "Siti Aminah", "keahlian": "Listrik & Instalasi",
         "rating": 4.5, "jumlah_review": 8, "pengalaman": "7 tahun pengalaman",
         "foto": "https://placehold.co/150x150"},
        {"id": 3, "nama": "Agus Wijaya", "keahlian": "Pemasangan Keramik",
         "rating": 4.7, "jumlah_review": 10, "pengalaman": "5 tahun pengalaman",
         "foto": "https://placehold.co/150x150"},
    ]
    tukang = next((t for t in simulated_tukang if t["id"] == tukang_id), None)
    
    if not tukang:
        flash("Tukang tidak ditemukan.", "warning")
        return redirect(url_for("rekomendasi"))
    
    return render_template("lihat_tukang.html", tukang=tukang)

@app.route("/chat")
def chat():
    tukang = {
        "nama": "Tukang Contoh",
        "telepon": "081234567890",
    }
    return render_template("chat.html", tukang=tukang)

@app.route("/profil_user")
def profil_user():
    customer = {
        "nama": "Andi Pratama",
        "telepon": "081234567890",
        "alamat": "Jl. Merdeka No. 10, Semarang"
    }
    return render_template("profil_user.html", customer=customer)

@app.route("/notifikasi")
def notifikasi():
    notifications = [
        {
            "pesan": "Tukang Budi membalas pesan Anda",
            "detail": "“Baik, saya bisa datang besok pagi.”",
            "tanggal": "08/11/2025"
        },
        {
            "pesan": "Pesanan Anda telah dikirim ke Tukang Siti Aminah",
            "detail": "Layanan: Instalasi Listrik Rumah",
            "tanggal": "07/11/2025"
        },
        {
            "pesan": "Tukang Agus mengirim pesan baru",
            "detail": "“Apakah warna keramiknya putih polos?”",
            "tanggal": "06/11/2025"
        }
    ]
    return render_template("notifikasi.html", notifications=notifications, active_page="notifikasi")

#HALAMAN ADMIN KELOLA CUSTOMER

@app.route('/admin/customers')
def kelola_customers():
    cursor.execute("SELECT * FROM users WHERE role = 'customer'")
    customers = cursor.fetchall()

    # Jika request minta JSON (untuk Postman)
    if request.args.get('json') == 'true':
        return jsonify(customers)

    # Default: tampilkan HTML
    return render_template('admin/customers.html', customers=customers)

from flask import request, jsonify, render_template, redirect, url_for, flash

@app.route('/admin/customers/add', methods=['GET', 'POST'])
def add_customer():
    if request.method == 'POST':
        # Jika request JSON (Postman)
        if request.is_json:
            data = request.get_json()
            username = data.get('username')
            email    = data.get('email')
            password = data.get('password')
        else:
            # Request dari form HTML
            username = request.form.get('username')
            email    = request.form.get('email')
            password = request.form.get('password')

        # Insert ke database
        cursor.execute("""
            INSERT INTO users (username, email, password, role)
            VALUES (%s, %s, %s, 'customer')
        """, (username, email, password))
        db.commit()

        # Respon untuk JSON
        if request.is_json:
            return jsonify({"message": "Customer berhasil ditambahkan!"}), 201

        # Respon untuk HTML
        flash("Customer berhasil ditambahkan!", "success")
        return redirect(url_for('kelola_customers'))

    # Jika GET → tampilkan form add_customer.html
    return render_template('admin/add_customer.html')

@app.route('/admin/customers/edit/<int:id>', methods=['GET', 'POST', 'PUT'])
def edit_customer(id):
    # Ambil data customer dari database
    cursor.execute("SELECT * FROM users WHERE id_users=%s", (id,))
    customer = cursor.fetchone()

    if request.method == 'POST' or request.method == 'PUT':
        # Jika request dari Postman (JSON)
        if request.is_json:
            data = request.get_json()
            username = data.get('username')
            email    = data.get('email')
        else:
            # Request dari form HTML
            username = request.form.get('username')
            email    = request.form.get('email')

        # Update database
        cursor.execute("""
            UPDATE users 
            SET username=%s, email=%s 
            WHERE id_users=%s
        """, (username, email, id))
        db.commit()

        # Respon untuk JSON
        if request.is_json:
            return jsonify({"message": "Customer berhasil diperbarui!"})

        # Respon untuk HTML
        flash("Customer berhasil diperbarui!", "success")
        return redirect(url_for('kelola_customers'))

    # Jika GET → tampilkan halaman edit_customer.html
    return render_template('admin/edit_customer.html', customer=customer)


@app.route('/admin/customers/delete/<int:id>', methods=['GET', 'DELETE'])
def delete_customer(id):
    # Hapus data customer
    cursor.execute("DELETE FROM users WHERE id_users=%s", (id,))
    db.commit()

    # Respon untuk Postman (DELETE)
    if request.method == 'DELETE':
        return jsonify({"message": "Customer berhasil dihapus!"})

    # Respon untuk browser (GET)
    flash("Customer berhasil dihapus!", "success")
    return redirect(url_for('kelola_customers'))

#HALAMAN ADMIN KELOLA TUKANG

@app.route('/admin/tukang')
def kelola_tukang():
    cursor.execute("SELECT * FROM users WHERE role = 'tukang'")
    tukang = cursor.fetchall()

    # Jika request minta JSON (untuk Postman)
    if request.args.get('json') == 'true':
        return jsonify(tukang)

    # Default: tampilkan HTML
    return render_template('admin/tukang.html', tukang=tukang, form_type=None)

@app.route('/admin/tukang/add', methods=['GET', 'POST'])
def add_tukang():

    # Jika request POST dan dari Postman (JSON)
    if request.method == 'POST' and request.is_json:
        data = request.get_json()
        username = data.get('username')
        email    = data.get('email')
        password = data.get('password')

        cursor.execute("""
            INSERT INTO users (username, email, password, role)
            VALUES (%s, %s, %s, 'tukang')
        """, (username, email, password))
        db.commit()

        return jsonify({"message": "Tukang berhasil ditambahkan!"}), 201

    # Jika POST dari Form HTML
    if request.method == 'POST':
        username = request.form['username']
        email    = request.form['email']
        password = request.form['password']

        cursor.execute("""
            INSERT INTO users (username, email, password, role)
            VALUES (%s, %s, %s, 'tukang')
        """, (username, email, password))
        db.commit()

        return redirect('/admin/tukang')

    # Jika GET → tampilkan form HTML
    return render_template("admin/add_tukang.html", tukang=[], form_type="add", data=None)

@app.route('/admin/tukang/edit/<int:id>', methods=['GET', 'POST', 'PUT'])
def edit_tukang(id):

    # Ambil data tukang berdasarkan ID
    cursor.execute("SELECT * FROM users WHERE id_users=%s", (id,))
    data = cursor.fetchone()

    # Jika request dari Postman (JSON)
    if request.method in ['POST', 'PUT'] and request.is_json:
        req = request.get_json()
        username = req.get('username')
        email    = req.get('email')

        cursor.execute("""
            UPDATE users 
            SET username=%s, email=%s 
            WHERE id_users=%s
        """, (username, email, id))
        db.commit()

        return jsonify({"message": "Tukang berhasil diperbarui!"})

    # Jika request POST dari HTML form
    if request.method == 'POST':
        username = request.form['username']
        email    = request.form['email']

        cursor.execute("""
            UPDATE users 
            SET username=%s, email=%s 
            WHERE id_users=%s
        """, (username, email, id))
        db.commit()

        return redirect('/admin/tukang')

    # Jika GET → tampilkan form edit HTML
    return render_template("admin/edit_tukang.html", tukang=[], form_type="edit", data=data)

@app.route('/admin/tukang/delete/<int:id>', methods=['GET', 'DELETE'])
def delete_tukang(id):
    cursor.execute("DELETE FROM users WHERE id_users=%s", (id,))
    db.commit()

    if request.method == 'GET':
        flash("Tukang berhasil dihapus", "success")
        return redirect('/admin/tukang')

    # Jika method DELETE (Postman)
    return jsonify({
        "message": "Tukang berhasil dihapus"
    })

@app.route('/logout')
def logout():
    session.clear()
    flash("Anda telah logout.")
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
