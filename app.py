from flask import Flask, render_template, redirect, url_for, jsonify, request, session, flash
import mysql.connector
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from functools import wraps
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import joblib

app = Flask(__name__)
app.config['SECRET_KEY'] = 'kunci-rahasia-teman-tukang-yang-kuat'

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="capstone_web"
)
cursor = db.cursor(dictionary=True)

# model cnn
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
 
# model review
svm_model = joblib.load('model/svm_model.pkl')
tfidf = joblib.load('model/tfidf_vectorizer.pkl')

def predict_sentiment(text):
    text_tfidf = tfidf.transform([text])
    result = svm_model.predict(text_tfidf)[0]

    return "positif" if result == 1 else "negatif"

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Anda harus login terlebih dahulu.", "warning")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# cbf (content based filtering - rekomendasi perhitungan)
cursor.execute("SELECT * FROM tukang")
TUKANG_DATA = cursor.fetchall() 

dokumen_tukang = [f"{t['keahlian']} {t['pengalaman']}" for t in TUKANG_DATA]

vectorizer = TfidfVectorizer()
TFIDF_MATRIX = vectorizer.fit_transform(dokumen_tukang)

print("TF-IDF tukang berhasil di-load (cached)")

# api 
@app.route("/api/rekomendasi")
def api_rekomendasi():
    jenis_kerusakan = request.args.get('jenis', None)

    if not jenis_kerusakan:
        return jsonify({"status": "error", "message": "Jenis kerusakan tidak ditemukan"}), 400

    query_vec = vectorizer.transform([jenis_kerusakan])
    sim_scores = cosine_similarity(query_vec, TFIDF_MATRIX).flatten()

    rekomendasi_list = []
    for i, t in enumerate(TUKANG_DATA):
        if sim_scores[i] >= 0.1:
            rekomendasi_list.append({
                "id_tukang": t["id_tukang"],
                "nama": t["nama"],
                "keahlian": t["keahlian"],
                "pengalaman": t["pengalaman"],
                "foto": t.get("foto", "https://placehold.co/80x80"),
                "similarity": float(sim_scores[i])
            })

    rekomendasi_list = sorted(rekomendasi_list, key=lambda x: x["similarity"], reverse=True)

    return jsonify({"status": "success", "rekomendasi": rekomendasi_list})

@app.route('/api/review', methods=['POST'])
def add_review():
    try:
        data = request.get_json()

        user_id = data.get('user_id')
        tukang_id = data.get('tukang_id')
        review_text = data.get('review_text')
        rating = data.get('rating')

        if user_id is None or tukang_id is None or not review_text or rating is None:
            return jsonify({"status": "error", "message": "Data tidak lengkap"}), 400

        rating = int(rating)

        sentiment = predict_sentiment(review_text)

        cursor.execute("""
            INSERT INTO review (user_id, tukang_id, review_text, sentiment, rating)
            VALUES (%s, %s, %s, %s, %s)
        """, (user_id, tukang_id, review_text, sentiment, rating))

        cursor.execute("""
            UPDATE tukang
            SET 
                rating = (
                    SELECT IFNULL(AVG(rating), 0)
                    FROM review WHERE tukang_id=%s
                ),
                jumlah_ulasan = (
                    SELECT COUNT(*) FROM review WHERE tukang_id=%s
                )
            WHERE id_tukang=%s
        """, (tukang_id, tukang_id, tukang_id))

        db.commit()

        return jsonify({
            "status": "success",
            "sentiment": sentiment
        }), 201

    except Exception as e:
        db.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500
    
# route admin
@app.route('/admin')
def admin_dashboard():
    if 'user_role' not in session or session['user_role'] != 'admin':
        flash("Akses ditolak!", "danger")
        return redirect(url_for('login_admin'))

    cursor.execute("SELECT COUNT(*) AS total FROM tukang")
    total_tukang = cursor.fetchone()['total']
    
    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM users
        WHERE role = 'customer'
    """)
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

# kelola customer
@app.route('/admin/customers')
def kelola_customers():
    cursor.execute("SELECT * FROM users WHERE role = 'customer'")
    customers = cursor.fetchall()

    if request.args.get('json') == 'true':
        return jsonify(customers)

    return render_template('admin/customers.html', customers=customers)

@app.route('/admin/customers/add', methods=['GET', 'POST'])
def add_customer():
    if request.method == 'POST':
        if request.is_json:
            data = request.get_json()
            username = data.get('username')
            email = data.get('email')
            password = data.get('password')
        else:
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')

        cursor.execute(
            """
            INSERT INTO users (username, email, password, role)
            VALUES (%s, %s, %s, 'customer')
            """,
            (username, email, password)
        )
        db.commit()

        if request.is_json:
            return jsonify({"message": "Customer berhasil ditambahkan!"}), 201

        flash("Customer berhasil ditambahkan!", "success")
        return redirect(url_for('kelola_customers'))

    return render_template('admin/add_customer.html')

@app.route('/admin/customers/edit/<int:id>', methods=['GET', 'POST'])
def edit_customer(id):
    cursor.execute("SELECT * FROM users WHERE id_users=%s", (id,))
    customer = cursor.fetchone()

    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        cursor.execute(
            "UPDATE users SET username=%s, email=%s, password=%s WHERE id_users=%s",
            (username, email, password, id)
        )
        db.commit()

        return redirect('/admin/customers')

    return render_template('admin/edit_customer.html', customer=customer)

@app.route('/admin/customers/delete/<int:id>', methods=['GET', 'DELETE'])
def delete_customer(id):
    cursor.execute("DELETE FROM users WHERE id_users=%s", (id,))
    db.commit()

    if request.method == 'DELETE':
        return jsonify({"message": "Customer berhasil dihapus!"})

    flash("Customer berhasil dihapus!", "success")
    return redirect(url_for('kelola_customers'))

# kelola tukang
@app.route('/admin/tukang')
def kelola_tukang():
    cursor.execute("SELECT * FROM tukang")
    tukang = cursor.fetchall()
    return render_template('admin/tukang.html', tukang=tukang)


@app.route('/admin/tukang/add', methods=['GET','POST'])
def add_tukang():
    if request.method == 'POST':
        nama = request.form['nama']
        keahlian = request.form['keahlian']
        pengalaman = request.form['pengalaman']
        foto = request.form['foto']

        cursor.execute("""
            INSERT INTO tukang (nama, keahlian, pengalaman, foto, rating)
            VALUES (%s,%s,%s,%s,0)
        """,(nama,keahlian,pengalaman,foto))
        db.commit()

        flash("Tukang berhasil ditambahkan","success")
        return redirect('/admin/tukang')

    return render_template('admin/add_tukang.html')


@app.route('/admin/tukang/edit/<int:id>', methods=['GET','POST'])
def edit_tukang(id):
    cursor.execute("SELECT * FROM tukang WHERE id_tukang=%s",(id,))
    tukang = cursor.fetchone()

    if not tukang:
        flash("Tukang tidak ditemukan","danger")
        return redirect('/admin/tukang')

    if request.method == 'POST':
        nama = request.form['nama']
        keahlian = request.form['keahlian']
        pengalaman = request.form['pengalaman']
        foto = request.form['foto']

        cursor.execute("""
            UPDATE tukang SET
            nama=%s, keahlian=%s, pengalaman=%s, foto=%s
            WHERE id_tukang=%s
        """,(nama,keahlian,pengalaman,foto,id))
        db.commit()

        flash("Tukang berhasil diupdate","success")
        return redirect('/admin/tukang')

    return render_template('admin/edit_tukang.html', t=tukang)


@app.route('/admin/tukang/delete/<int:id>')
def delete_tukang(id):
    cursor.execute("DELETE FROM tukang WHERE id_tukang=%s",(id,))
    db.commit()
    flash("Tukang berhasil dihapus","success")
    return redirect('/admin/tukang')

# route tampilan customer
@app.route('/')
def index():
    return redirect(url_for('dashboard'))

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
@login_required
def tulis_ulasan(order_id):

    cursor.execute(
        "SELECT tukang_id FROM orders WHERE id_order=%s",
        (order_id,)
    )
    order = cursor.fetchone()

    if not order:
        flash("Pesanan tidak ditemukan", "danger")
        return redirect(url_for('riwayat_pesanan'))

    tukang_id = order['tukang_id']
    user_id = session['user_id']

    if request.method == 'POST':
        rating = request.form['rating']
        review_text = request.form['ulasan']

        sentiment = predict_sentiment(review_text)

        cursor.execute("""
            INSERT INTO review (user_id, tukang_id, review_text, sentiment, rating)
            VALUES (%s, %s, %s, %s, %s)
        """, (user_id, tukang_id, review_text, sentiment, rating))

        cursor.execute("""
            UPDATE tukang
            SET 
                rating = (
                    SELECT IFNULL(AVG(rating), 0)
                    FROM review WHERE tukang_id=%s
                ),
                jumlah_ulasan = (
                    SELECT COUNT(*) FROM review WHERE tukang_id=%s
                )
            WHERE id_tukang=%s
        """, (tukang_id, tukang_id, tukang_id))

        db.commit()

        flash("Ulasan berhasil dikirim", "success")
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

@app.route('/booking', methods=['GET', 'POST'])
def booking():
    if 'user_id' not in session:
        flash("Silakan login terlebih dahulu.", "warning")
        return redirect(url_for('login'))

    if request.method == 'POST':
        tanggal = request.form.get('date', '')
        waktu = request.form.get('time', '')
        opsi = request.form.get('price_option', '')
        custom = request.form.get('custom_price', '')

        try:
            if opsi == 'custom':
                harga = int(custom) if custom else 0
            else:
                harga = int(opsi) if opsi else 0
        except ValueError:
            flash("Masukkan angka yang valid.", "danger")
            return redirect(url_for('booking'))

        flash(f"Terpilih: {tanggal} {waktu} — Rp {harga:,}", "success")
        return redirect(url_for('booking'))

    return render_template('booking.html')

@app.route("/rekomendasi")
def rekomendasi():
    jenis_kerusakan = request.args.get('jenis', None)

    if not jenis_kerusakan:
        flash("Jenis kerusakan tidak ditemukan.", "warning")
        return redirect(url_for('dashboard'))

    query_vec = vectorizer.transform([jenis_kerusakan])
    sim_scores = cosine_similarity(query_vec, TFIDF_MATRIX).flatten()

    rekomendasi_list = []
    for i, t in enumerate(TUKANG_DATA):
        if sim_scores[i] >= 0.1:  
            rekomendasi_list.append({
                "id_tukang": t["id_tukang"],
                "nama": t["nama"],
                "keahlian": t["keahlian"],
                "pengalaman": t["pengalaman"],
                "foto": t.get("foto", "https://placehold.co/80x80"),
                "similarity": float(sim_scores[i])
            })

    rekomendasi_list = sorted(rekomendasi_list, key=lambda x: x["similarity"], reverse=True)

    return render_template("rekomendasi.html", tukangs=rekomendasi_list, jenis=jenis_kerusakan)

@app.route("/lihat-tukang/<int:tukang_id>")
def lihat_tukang(tukang_id):
    cursor.execute("SELECT * FROM tukang WHERE id_tukang=%s", (tukang_id,))
    tukang = cursor.fetchone()

    if not tukang:
        flash("Tukang tidak ditemukan.", "warning")
        return redirect(url_for("rekomendasi"))

    tukang.setdefault("rating", 0)
    tukang.setdefault("jumlah_ulasan", 0)
    tukang.setdefault("foto", "https://placehold.co/150x150")

    pengalaman_str = tukang.get("pengalaman", "")
    tukang["pengalaman"] = [x.strip() for x in pengalaman_str.split(",") if x.strip()]
    cursor.execute("""
        SELECT 
            r.review_text,
            r.rating,
            r.sentiment,
            u.username AS nama
        FROM review r
        JOIN users u ON r.user_id = u.id_users
        WHERE r.tukang_id = %s
        ORDER BY r.tanggal DESC
    """, (tukang_id,))
    
    reviews = cursor.fetchall()

    total_ulasan = len(reviews)
    negatif = sum(1 for r in reviews if r["sentiment"] == "negatif")

    persentase_negatif = 0
    if total_ulasan > 0:
        persentase_negatif = round((negatif / total_ulasan) * 100)

    tukang["persentase_negatif"] = persentase_negatif
    tukang["total_ulasan"] = total_ulasan
   
    tukang["ulasan"] = [
        {
            "nama": r["nama"],
            "rating": r["rating"],
            "komentar": r["review_text"],
            "sentiment": r["sentiment"],
            "foto": "https://placehold.co/55x55"
        }
        for r in reviews
    ]

    return render_template("lihat_tukang.html", tukang=tukang)

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

@app.route('/logout')
def logout():
    session.clear()
    flash("Anda telah logout.")
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
