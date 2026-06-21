 # Human Capture — Flask Dashboard

Real-time face recognition door monitoring system with web dashboard.

## Quick Start

```bash
# 1. Install dependencies
pip install flask opencv-python face_recognition numpy requests

# 2. Add known face photos
#    Drop photos in known_faces/ folder
#    Name them: abu_bakar.jpg, person2.jpg etc

# 3. (Optional) Add Telegram credentials in app.py
#    BOT_TOKEN = "your_token"
#    CHAT_ID   = "your_chat_id"

# 4. Run
python app.py

# 5. Open browser
http://localhost:5000
```

## Features
- Live camera feed in browser
- Real-time face detection & recognition
- Alert log with snapshots
- Add/remove people from dashboard
- Telegram notifications with photo
- Stats: today's count, total, known faces

## Switch to IP Camera
In app.py line ~80, change:
```python
cap = cv2.VideoCapture(0)
# to:
cap = cv2.VideoCapture("rtsp://user:pass@192.168.1.x:554/stream")
```

## Project Structure
```
human_capture/
├── app.py              ← Main Flask app + camera thread
├── templates/
│   └── index.html      ← Full dashboard UI
├── known_faces/        ← Drop face photos here
├── snapshots/          ← Auto-saved detection images
└── requirements.txt
```
