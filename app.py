  """
 Human Capture — Flask Dashboard
Notifications : ntfy.sh
DB            : SQLite
Run           : python app.py
Open          : http://localhost:5000
"""
 
import os, time, threading, sqlite3, json, sys
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
from datetime import datetime, date
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
import cv2
import numpy as np

# ── face_recognition ──────────────────────────────────────────
try:
    import face_recognition
    FACE_REC = True
except ImportError:
    FACE_REC = False
    print("⚠  face_recognition not installed — DEMO MODE")

# ── requests ──────────────────────────────────────────────────
try:
    import requests as http
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("⚠  requests not installed — pip install requests")

# ══════════════════════════════════════════════════════════════
#  CONFIG  ← only edit this section
# ══════════════════════════════════════════════════════════════
NTFY_TOPIC      = "humancapture_abubakar_9321"  # must match what you subscribed to in ntfy app
KNOWN_FACES_DIR = "known_faces"
SNAPSHOTS_DIR   = "snapshots"
DB_PATH         = "human_capture.db"
COOLDOWN        = 30      # seconds before same person triggers again
TOLERANCE       = 0.55    # lower = stricter. increase to 0.6 if missing detections
# ══════════════════════════════════════════════════════════════

app = Flask(__name__)
app.secret_key = "hc2026"

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(SNAPSHOTS_DIR,   exist_ok=True)

# ── Shared state ──────────────────────────────────────────────
_lock           = threading.Lock()
known_encodings = []
known_names     = []
last_seen       = {}
camera_active   = False
cam_online      = False
latest_frame    = None
frame_lock      = threading.Lock()
face_rec_lock   = threading.Lock()

# In-memory counters — updated instantly on detection
_today_count    = 0
_total_count    = 0
_last_alert     = "—"

# ══════════════════════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════════════════════

def get_db():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    c = get_db()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS detections (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT NOT NULL,
            timestamp  TEXT NOT NULL,
            snapshot   TEXT,
            confidence REAL DEFAULT 0.0
        );
        CREATE TABLE IF NOT EXISTS people (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            name     TEXT NOT NULL,
            filename TEXT NOT NULL UNIQUE,
            added_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS i1 ON detections(timestamp);
        CREATE INDEX IF NOT EXISTS i2 ON detections(name);
    """)
    c.commit(); c.close()
    print("✅ DB ready")

def _load_counters():
    global _today_count, _total_count, _last_alert
    c   = get_db()
    td  = date.today().isoformat()
    _total_count = c.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
    _today_count = c.execute(
        "SELECT COUNT(*) FROM detections WHERE timestamp LIKE ?", (f"{td}%",)
    ).fetchone()[0]
    row = c.execute(
        "SELECT name,timestamp FROM detections ORDER BY id DESC LIMIT 1"
    ).fetchone()
    _last_alert = f"{row['name']}  {row['timestamp']}" if row else "—"
    c.close()

def db_insert(name, snapshot, confidence):
    global _today_count, _total_count, _last_alert
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c  = get_db()
    c.execute(
        "INSERT INTO detections (name,timestamp,snapshot,confidence) VALUES (?,?,?,?)",
        (name, ts, snapshot, confidence)
    )
    c.commit(); c.close()
    # Update in-memory instantly — no lag
    with _lock:
        _today_count += 1
        _total_count += 1
        _last_alert   = f"{name}  {ts}"
    return ts

def db_get_log(limit=100, name_f=None, date_f=None):
    c      = get_db()
    q      = "SELECT * FROM detections WHERE 1=1"
    params = []
    if name_f: q += " AND name LIKE ?";      params.append(f"%{name_f}%")
    if date_f: q += " AND timestamp LIKE ?"; params.append(f"{date_f}%")
    q += " ORDER BY id DESC LIMIT ?";        params.append(limit)
    rows = c.execute(q, params).fetchall()
    c.close()
    now = datetime.now()
    result = []
    for r in rows:
        d = dict(r)
        dt   = datetime.strptime(d["timestamp"], "%Y-%m-%d %H:%M:%S")
        diff = int((now - dt).total_seconds())
        d["time_ago"] = (f"{diff}s ago"       if diff < 60    else
                         f"{diff//60}m ago"   if diff < 3600  else
                         f"{diff//3600}h ago" if diff < 86400 else
                         f"{diff//86400}d ago")
        result.append(d)
    return result

def db_person_stats():
    c    = get_db()
    rows = c.execute(
        "SELECT name, COUNT(*) as count FROM detections GROUP BY name ORDER BY count DESC"
    ).fetchall()
    c.close()
    return [dict(r) for r in rows]

def db_get_people():
    c    = get_db()
    rows = c.execute("SELECT * FROM people ORDER BY added_at DESC").fetchall()
    c.close()
    return [dict(r) for r in rows]

def db_add_person(name, filename):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c  = get_db()
    c.execute("INSERT OR REPLACE INTO people (name,filename,added_at) VALUES (?,?,?)",
              (name, filename, ts))
    c.commit(); c.close()

def db_del_person(filename):
    c = get_db()
    c.execute("DELETE FROM people WHERE filename=?", (filename,))
    c.commit(); c.close()

def db_clear(mode="today"):
    global _today_count, _total_count
    c = get_db()
    if mode == "all":
        c.execute("DELETE FROM detections")
        with _lock: _today_count = _total_count = 0
    else:
        c.execute("DELETE FROM detections WHERE timestamp LIKE ?",
                  (f"{date.today().isoformat()}%",))
        with _lock: _today_count = 0
    c.commit(); c.close()

# ══════════════════════════════════════════════════════════════
#  FACE RECOGNITION
# ══════════════════════════════════════════════════════════════

def load_faces():
    global known_encodings, known_names
    if not FACE_REC: return
    with face_rec_lock:
        known_encodings, known_names = [], []
        for f in os.listdir(KNOWN_FACES_DIR):
            if not f.lower().endswith((".jpg",".jpeg",".png")): continue
            try:
                img  = face_recognition.load_image_file(os.path.join(KNOWN_FACES_DIR, f))
                encs = face_recognition.face_encodings(img)
                if encs:
                    known_encodings.append(encs[0])
                    known_names.append(os.path.splitext(f)[0].replace("_"," ").title())
                    print(f"   ✓ {known_names[-1]}")
                else:
                    print(f"   ✗ No face in {f} — use clearer photo")
            except Exception as e:
                print(f"   ✗ {f}: {e}")
        print(f"Faces loaded: {known_names or 'NONE'}")

# ══════════════════════════════════════════════════════════════
#  NTFY  — sends text alert + photo
# ══════════════════════════════════════════════════════════════

def ntfy_notify(name, snap_path, ts):
    if not HAS_REQUESTS:
        print("   ⚠  pip install requests  — needed for ntfy"); return
    if not NTFY_TOPIC:
        print("   ⚠  Set NTFY_TOPIC in app.py"); return

    url = f"https://ntfy.sh/{NTFY_TOPIC}"

    # 1️⃣  Text notification
    try:
        r = http.post(url,
            data    = f"User {name} detected at {ts}".encode("utf-8"),
            headers = {
                "Title":    f"Human Capture - {name}",
                "Priority": "high",
                "Tags":     "rotating_light",
            },
            timeout = 10
        )
        print(f"   📱 ntfy text → {r.status_code}")
    except Exception as e:
        print(f"   ❌ ntfy text error: {e}")

    # 2️⃣  Photo notification
    try:
        if os.path.exists(snap_path):
            with open(snap_path, "rb") as f:
                data = f.read()
            r2 = http.post(url,
                data    = data,
                headers = {
                    "Title":        f"Photo - {name}",
                    "Filename":     os.path.basename(snap_path),
                    "Content-Type": "image/jpeg",
                    "Tags":         "camera",
                },
                timeout = 15
            )
            print(f"   🖼  ntfy photo → {r2.status_code}")
    except Exception as e:
        print(f"   ❌ ntfy photo error: {e}")

# ══════════════════════════════════════════════════════════════
#  CAMERA THREAD
# ══════════════════════════════════════════════════════════════

def camera_thread():
    global latest_frame, camera_active, cam_online
    print("Connecting to local webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Local webcam not found")
        camera_active = cam_online = False
        return

    camera_active = cam_online = True
    print("📷 Camera ON\n")

    while camera_active:
        ret, frame = cap.read()
        if not ret: time.sleep(0.1); continue

        display = frame.copy()

        if FACE_REC and known_encodings:
            small = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            with face_rec_lock:
                locs  = face_recognition.face_locations(rgb)
                encs  = face_recognition.face_encodings(rgb, locs)

            for enc, loc in zip(encs, locs):
                dists = face_recognition.face_distance(known_encodings, enc)
                name  = "Unknown"
                conf  = 0.0

                print(f"   dist={[round(d,3) for d in dists]}  tol={TOLERANCE}")

                if len(dists) and min(dists) < TOLERANCE:
                    idx  = int(np.argmin(dists))
                    name = known_names[idx]
                    conf = round(1 - float(dists[idx]), 3)
                    print(f"   ✅ {name} ({conf:.0%})")
                else:
                    mn = round(min(dists),3) if len(dists) else "—"
                    print(f"   ❓ Unknown  closest={mn}  (increase TOLERANCE if this is you)")

                now = time.time()
                if name != "Unknown":
                    if name not in last_seen or (now - last_seen[name]) > COOLDOWN:
                        last_seen[name] = now
                        fn   = f"{name.replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        sp   = os.path.join(SNAPSHOTS_DIR, fn)
                        cv2.imwrite(sp, frame)
                        ts   = db_insert(name, fn, conf)
                        print(f"   💾 Saved  today={_today_count}  total={_total_count}")
                        threading.Thread(target=ntfy_notify,
                                         args=(name, sp, ts), daemon=True).start()
                    else:
                        left = int(COOLDOWN-(now-last_seen[name]))
                        print(f"   ⏳ Cooldown {left}s")

                # Draw box
                t,r,b,l = [v*4 for v in loc]
                col = (0,220,80) if name != "Unknown" else (0,0,220)
                cv2.rectangle(display,(l,t),(r,b),col,2)
                cv2.rectangle(display,(l,b-30),(r,b),col,-1)
                if name != "Unknown":
                    cv2.putText(display,"DETECTED",(l,t-28),
                                cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,120),2)
                lbl = f"{name} {conf:.0%}" if name!="Unknown" else "Unknown"
                cv2.putText(display,lbl,(l+5,b-8),
                            cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1)

        elif FACE_REC and not known_encodings:
            cv2.putText(display,"No faces registered — add via dashboard",
                        (10,32),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,200,255),2)
        else:
            cv2.putText(display,"DEMO MODE — face_recognition not installed",
                        (10,32),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,200,255),2)

        cv2.putText(display, datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
                    (10,display.shape[0]-8),
                    cv2.FONT_HERSHEY_SIMPLEX,0.42,(140,140,140),1)

        with frame_lock:
            latest_frame = display.copy()

        time.sleep(0.04)

    cap.release()
    cam_online = camera_active = False
    print("📷 Camera OFF")

def gen_stream():
    while True:
        with frame_lock: frame = latest_frame
        if frame is None:
            frame = np.zeros((360,640,3),dtype=np.uint8)
            cv2.putText(frame,"Camera Offline — Press START",
                        (130,185),cv2.FONT_HERSHEY_SIMPLEX,0.65,(70,70,70),1)
        _, jpg = cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,72])
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
        time.sleep(0.04)

# ══════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════

@app.route("/")
def index(): return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/stats")
def api_stats():
    with _lock:
        return jsonify({
            "today":      _today_count,
            "total":      _total_count,
            "last_alert": _last_alert,
            "online":     cam_online,
            "known":      len(known_names)
        })

@app.route("/api/log")
def api_log():
    return jsonify(db_get_log(
        limit  = int(request.args.get("limit",100)),
        name_f = request.args.get("name") or None,
        date_f = request.args.get("date") or None
    ))

@app.route("/api/person_stats")
def api_person_stats(): return jsonify(db_person_stats())

@app.route("/api/people")
def api_people(): return jsonify(db_get_people())

@app.route("/api/add_person", methods=["POST"])
def api_add_person():
    if "photo" not in request.files or not request.form.get("name"):
        return jsonify({"error":"Name and photo required"}), 400
    name  = request.form["name"].strip()
    photo = request.files["photo"]
    ext   = os.path.splitext(photo.filename)[1].lower() or ".jpg"
    fname = name.replace(" ","_").lower() + ext
    path  = os.path.join(KNOWN_FACES_DIR, fname)
    photo.save(path)
    if FACE_REC:
        try:
            with face_rec_lock:
                encs = face_recognition.face_encodings(face_recognition.load_image_file(path))
            if not encs:
                os.remove(path)
                return jsonify({"error":"No face found. Use a clear front-facing photo."}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    db_add_person(name, fname)
    load_faces()
    return jsonify({"success": True, "message": f"{name} registered"})

@app.route("/api/remove_person", methods=["POST"])
def api_remove_person():
    fname = request.json.get("file","")
    p = os.path.join(KNOWN_FACES_DIR, fname)
    if os.path.exists(p): os.remove(p)
    db_del_person(fname)
    load_faces()
    return jsonify({"success": True})

@app.route("/api/camera/start", methods=["POST"])
def api_cam_start():
    global camera_active
    if not camera_active:
        threading.Thread(target=camera_thread, daemon=True).start()
        time.sleep(0.6)
    return jsonify({"ok": True})

@app.route("/api/camera/stop", methods=["POST"])
def api_cam_stop():
    global camera_active
    camera_active = False
    return jsonify({"ok": True})

@app.route("/api/clear_log", methods=["POST"])
def api_clear():
    db_clear(request.json.get("mode","today"))
    return jsonify({"ok": True})

@app.route("/snapshots/<fn>")
def snap(fn): return send_from_directory(SNAPSHOTS_DIR, fn)

@app.route("/audio/<fn>")
def serve_audio(fn):
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), fn)

# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  HUMAN CAPTURE")
    print("="*50)
    init_db()
    _load_counters()
    print(f"  Counts  today={_today_count}  total={_total_count}")
    print(f"\n  Loading faces...")
    load_faces()
    print(f"\n  ntfy  → ntfy.sh/{NTFY_TOPIC}")
    print(f"  Tol   → {TOLERANCE}  |  Cooldown → {COOLDOWN}s")
    print(f"\n  Dashboard → http://localhost:5000")
    print("="*50 + "\n")
    threading.Thread(target=camera_thread, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
