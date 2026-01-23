import os
from app import app

# Gunicorn will import `app` from this module (wsgi:app)
# Optional: allow running with `python wsgi.py` for local debug
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
