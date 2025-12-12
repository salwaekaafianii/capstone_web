from flask import (
    Flask, render_template, redirect, url_for, jsonify,
    request, session, flash
)
import mysql.connector
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)
app.config['SECRET_KEY'] = 'kunci-rahasia-teman-tukang-yang-kuat'

# --- KONEKSI DATABASE ---
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="capstone_web"
)
cursor = db.cursor(dictionary=True)

cursor.execute("SELECT * FROM tukang")
TUKANG_DATA = cursor.fetchall()  # Disimpan global, tidak query ulang

# Buat dokumen untuk TF-IDF
dokumen_tukang = [f"{t['keahlian']} {t['pengalaman']}" for t in TUKANG_DATA]

# Siapkan vectorizer & matrix (hanya sekali)
vectorizer = TfidfVectorizer()
TFIDF_MATRIX = vectorizer.fit_transform(dokumen_tukang)

print("TF-IDF tukang berhasil di-load (cached)")

from functools import wraps

# --- LOAD MODEL CNN ---
model = load_model("model/model_temantukang.keras")
labels = ["Retak Dinding", "Plafon Rusak", "Keramik Rusak", "Cat Mengelupas", "Kayu Kusen Lapuk", "Dinding Berjamur"]

analisis_faktor = {
    "Retak Dinding": "Kerusakan terjadi karena fondasi mengalami penurunan tidak merata, getaran berulang, atau tekanan beban berlebih pada struktur dinding.",
    "Plafon Rusak": "Kerusakan plafon biasanya disebabkan oleh kebocoran atap, rembesan air AC, atau material plafon yang sudah rapuh dan tidak mampu menahan beban.",
    "Keramik Rusak": "Keramik retak atau terangkat dapat terjadi akibat permukaan lantai yang tidak rata, penurunan tanah, atau pemasangan awal yang kurang tepat.",
    "Cat Mengelupas": "Cat mengelupas umumnya dipicu oleh kelembaban tinggi, rembesan air, atau permukaan dinding yang tidak dibersihkan dengan baik sebelum pengecatan.",
    "Kayu Kusen Lapuk": "Kusen kayu dapat lapuk karena paparan air, kelembaban tinggi, atau serangan jamur dan rayap, sehingga kayu kehilangan kekuatan strukturalnya.",
    "Dinding Berjamur": "Dinding berjamur terjadi akibat kelembaban berlebih, ventilasi yang buruk, atau rembesan air yang terus-menerus, sehingga jamur berkembang di permukaan dinding.",
}

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Anda harus login terlebih dahulu.", "warning")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# --- ROUTES UTAMA ---
@app.route('/')
def index():
    return redirect(url_for('dashboard'))
@app.route('/admin')
def admin_dashboard():
    if 'user_role' not in session or session['user_role'] != 'admin':
        flash("Akses ditolak!", "danger")
        return redirect(url_for('login'))

    cursor.execute("SELECT COUNT(*) AS total FROM users WHERE role='tukang'")
    total_tukang = cursor.fetchone()['total']

    cursor.execute("SELECT COUNT(*) AS total FROM users WHERE role='customer'")
    total_customer = cursor.fetchone()['total']

    return render_template(
        'admin/admin_dashboard.html',
        total_tukang=total_tukang,
        total_customer=total_customer
    )
    
@app.route('/login/admin', methods=['GET', 'POST'])
def login_admin():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        cursor.execute("SELECT * FROM users WHERE email=%s AND role='admin'", (email,))
        admin = cursor.fetchone()

        if not admin:
            flash("Admin tidak ditemukan!", "danger")
            return redirect(url_for('login_admin'))

        if admin['password'] != password:
            flash("Password salah!", "danger")
            return redirect(url_for('login_admin'))

        session['user_id'] = admin['id_users']
        session['user_role'] = admin['role']
        session['user_email'] = admin['email']

        return redirect(url_for('admin_dashboard'))

    return render_template('admin/login_admin.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()

        if not user:
            flash("Email tidak terdaftar!", "danger")
            return redirect(url_for('login'))

        # CEGAH ADMIN LOGIN DI ROUTE INI
        if user['role'] == 'admin':
            flash("Gunakan halaman login admin!", "danger")
            return redirect(url_for('login_admin'))

        if user['password'] != password:
            flash("Password salah!", "danger")
            return redirect(url_for('login'))

        session['user_id'] = user['id_users']
        session['user_email'] = user['email']
        session['user_role'] = user['role']

        flash("Login berhasil!", "success")

        if user['role'] == "tukang":
            return redirect(url_for('tukang_dashboard'))
        else:
            return redirect(url_for('dashboard'))

    return render_template('login.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        if len(password) < 6 or len(password) > 8:
            flash("Password harus 6-8 karakter!", "danger")
            return redirect(url_for('register'))

        cursor.execute(
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

@app.route('/artikel-kerusakan')
def artikel_kerusakan():
    return render_template('artikel-kerusakan.html')

@app.route('/artikel-renovasi')
def artikel_renovasi():
    return render_template('artikel-renovasi.html')

@app.route('/riwayat-pesanan')
@login_required
def riwayat_pesanan():
    if 'user_id' not in session:
        flash("Anda harus login untuk melihat riwayat pesanan.", "warning")
        return redirect(url_for('login'))

    simulated_orders = {
        101: {'layanan': 'Perbaikan Pipa Bocor', 'tukang': 'Ahmad Imam', 'tanggal': '15/10/2025', 'status': 'Menunggu'},
        102: {'layanan': 'Pemasangan Keramik', 'tukang': 'Ibu Rina', 'tanggal': '01/11/2025', 'status': 'Selesai'},
    }
    return render_template('riwayat_pesanan.html', orders=simulated_orders, active_page='riwayat_pesanan')

@app.route('/ulasan/<int:order_id>', methods=['GET', 'POST'])
def tulis_ulasan(order_id):
    if request.method == 'POST':
        return redirect(url_for('riwayat_pesanan'))
    return render_template('tulis_ulasan.html', order_id=order_id)

@app.route('/deteksi', methods=['GET', 'POST'])
@login_required
def deteksi():
    if 'user_id' not in session:
        flash("Anda harus login untuk menggunakan fitur deteksi.", "warning")
        return redirect(url_for('login'))

    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            flash("Pilih gambar terlebih dahulu!", "danger")
            return redirect(url_for('deteksi'))

        os.makedirs("static/uploads", exist_ok=True)
        filepath = os.path.join("static/uploads", file.filename)
        file.save(filepath)

        img = Image.open(filepath).convert("RGB")
        img = img.resize((128, 128))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)
        label_index = np.argmax(pred)
        confidence = float(np.max(pred) * 100)
        hasil = labels[label_index]

        return render_template(
            "deteksi_hasil.html",
            gambar=file.filename,
            hasil=hasil,
            confidence=round(confidence, 2),
            analisis=analisis_faktor.get(hasil, "Tidak ada analisis tersedia.")
        )

    return render_template("deteksi.html")

@app.route("/rekomendasi")
def rekomendasi():
    jenis_kerusakan = request.args.get('jenis', None)

    if not jenis_kerusakan:
        flash("Jenis kerusakan tidak ditemukan.", "warning")
        return redirect(url_for('dashboard'))

    # Hitung similarity
    query_vec = vectorizer.transform([jenis_kerusakan])
    sim_scores = cosine_similarity(query_vec, TFIDF_MATRIX).flatten()

    # Siapkan data untuk template, hanya yang similarity >= 0.5
    rekomendasi_list = []
    for i, t in enumerate(TUKANG_DATA):
        if sim_scores[i] >= 0.1:  # filter kemiripan >= 0.5
            rekomendasi_list.append({
                "id_tukang": t["id_tukang"],
                "nama": t["nama"],
                "keahlian": t["keahlian"],
                "pengalaman": t["pengalaman"],
                "foto": t.get("foto", "https://placehold.co/80x80"),
                "similarity": float(sim_scores[i])
            })

    # Urutkan berdasarkan similarity tertinggi
    rekomendasi_list = sorted(rekomendasi_list, key=lambda x: x["similarity"], reverse=True)

    return render_template("rekomendasi.html", tukangs=rekomendasi_list, jenis=jenis_kerusakan)

@app.route("/lihat-tukang/<int:tukang_id>")
def lihat_tukang(tukang_id):
    # Ambil data tukang sesuai id_tukang
    cursor.execute("SELECT * FROM tukang WHERE id_tukang=%s", (tukang_id,))
    tukang = cursor.fetchone()

    if not tukang:
        flash("Tukang tidak ditemukan.", "warning")
        return redirect(url_for("rekomendasi"))

    # Pastikan data rating dan jumlah_ulasan ada untuk template
    tukang.setdefault("rating", 0)
    tukang.setdefault("jumlah_ulasan", 0)
    tukang.setdefault("foto", "https://placehold.co/150x150")

    # --- PERBAIKI BAGIAN PENGALAMAN ---
    # Jika pengalaman ada dan berbentuk string, ubah jadi list
    pengalaman_str = tukang.get("pengalaman", "")
    tukang["pengalaman"] = [x.strip() for x in pengalaman_str.split(",") if x.strip()]

    return render_template("lihat_tukang.html", tukang=tukang)

# --- PROFIL & CHAT ---
@app.route("/chat")
@login_required
def chat():
    tukang = {"nama": "Tukang Contoh", "telepon": "081234567890"}
    return render_template("chat.html", tukang=tukang)

@app.route("/profil_user")
@login_required
def profil_user():
    customer = {"nama": "Andi Pratama", "telepon": "081234567890", "alamat": "Jl. Merdeka No. 10, Semarang"}
    return render_template("profil_user.html", customer=customer)

@app.route("/notifikasi")
@login_required
def notifikasi():
    notifications = [
        {"pesan": "Tukang Budi membalas pesan Anda", "detail": "“Baik, saya bisa datang besok pagi.”", "tanggal": "08/11/2025"},
        {"pesan": "Pesanan Anda telah dikirim ke Tukang Siti Aminah", "detail": "Layanan: Instalasi Listrik Rumah", "tanggal": "07/11/2025"},
        {"pesan": "Tukang Agus mengirim pesan baru", "detail": "“Apakah warna keramiknya putih polos?”", "tanggal": "06/11/2025"}
    ]
    return render_template("notifikasi.html", notifications=notifications, active_page="notifikasi")
# HALAMAN ADMIN KELOLA CUSTOMER
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
            email = data.get('email')
            password = data.get('password')
        else:
            # Request dari form HTML
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')

        # Insert ke database
        cursor.execute(
            """
            INSERT INTO users (username, email, password, role)
            VALUES (%s, %s, %s, 'customer')
            """,
            (username, email, password)
        )
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
            email = data.get('email')
        else:
            # Request dari form HTML
            username = request.form.get('username')
            email = request.form.get('email')

        # Update database
        cursor.execute(
            """
            UPDATE users 
            SET username=%s, email=%s 
            WHERE id_users=%s
            """,
            (username, email, id)
        )
        db.commit()

        # Respon untuk JSON
        if request.is_json:
            return jsonify({"message": "Customer berhasil diperbarui!"})

        # Respon untuk HTML
        flash("Customer berhasil diperbarui!", "success")
        return redirect(url_for('kelola_customers'))

    # Jika GET → tampilkan halaman edit_customer.html
    return render_template('admin/edit_customer.html', customer=customer)


@app.route('/admin/customers/update/<int:id>', methods=['PATCH'])
def patch_customer(id):
    data = request.get_json()

    # Ambil data lama
    cursor.execute("SELECT username, email FROM users WHERE id_users=%s", (id,))
    old = cursor.fetchone()

    if not old:
        return jsonify({"error": "Customer tidak ditemukan"}), 404

    username = data.get('username', old['username'])
    email = data.get('email', old['email'])

    cursor.execute(
        """
        UPDATE users SET username=%s, email=%s WHERE id_users=%s
        """,
        (username, email, id)
    )
    db.commit()

    return jsonify({"message": "Customer berhasil diupdate (PATCH)!"})


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
        email = data.get('email')
        password = data.get('password')

        cursor.execute(
            """
            INSERT INTO users (username, email, password, role)
            VALUES (%s, %s, %s, 'tukang')
            """,
            (username, email, password)
        )
        db.commit()

        return jsonify({"message": "Tukang berhasil ditambahkan!"}), 201

    # Jika POST dari Form HTML
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        cursor.execute(
            """
            INSERT INTO users (username, email, password, role)
            VALUES (%s, %s, %s, 'tukang')
            """,
            (username, email, password)
        )
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
        email = req.get('email')

        cursor.execute(
            """
            UPDATE users 
            SET username=%s, email=%s 
            WHERE id_users=%s
            """,
            (username, email, id)
        )
        db.commit()

        return jsonify({"message": "Tukang berhasil diperbarui!"})

    # Jika request POST dari HTML form
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']

        cursor.execute(
            """
            UPDATE users 
            SET username=%s, email=%s 
            WHERE id_users=%s
            """,
            (username, email, id)
        )
        db.commit()

        return redirect('/admin/tukang')

    # Jika GET → tampilkan form edit HTML
    return render_template("admin/edit_tukang.html", tukang=[], form_type="edit", data=data)


@app.route('/admin/tukang/update/<int:id>', methods=['PATCH'])
def patch_tukang(id):
    data = request.get_json()

    cursor.execute("SELECT username, email FROM users WHERE id_users=%s", (id,))
    old = cursor.fetchone()

    if not old:
        return jsonify({"error": "Tukang tidak ditemukan"}), 404

    username = data.get('username', old['username'])
    email = data.get('email', old['email'])

    cursor.execute(
        """
        UPDATE users SET username=%s, email=%s WHERE id_users=%s
        """,
        (username, email, id)
    )
    db.commit()

    return jsonify({"message": "Tukang berhasil diupdate (PATCH)!"})


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

# --- LOGOUT ---
@app.route('/logout')
def logout():
    session.clear()
    flash("Anda telah logout.")
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
