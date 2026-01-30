"""
download_emails_imap.py
Télécharge des emails via IMAP (z.imt.fr) et les sauvegarde dans TP4/data/emails/

- 1 email = 1 fichier Markdown
- Cache SQLite pour éviter les doublons
"""

import os
import re
import sqlite3
import imaplib
import email
from email import policy
from email.header import decode_header
from datetime import datetime, timedelta
from getpass import getpass


HOST = "z.imt.fr"
PORT = 993

DATA_DATA_DIR = os.path.join("data")
EMAIL_DIR = os.path.join(DATA_DIR, "emails")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
DB_PATH = os.path.join(CACHE_DIR, "emails_cache.sqlite")


def ensure_dirs():
    os.makedirs(EMAIL_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS downloaded_emails (
            account TEXT NOT NULL,
            message_id TEXT NOT NULL,
            folder TEXT,
            PRIMARY KEY (account, message_id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sync_status (
            account TEXT PRIMARY KEY,
            last_synced TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def was_downloaded(conn, account: str, message_id: str) -> bool:
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM downloaded_emails WHERE account=? AND message_id=?",
        (account, message_id),
    )
    return cur.fetchone() is not None


def mark_downloaded(conn, account: str, message_id: str, folder: str):
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO downloaded_emails(account, message_id, folder) VALUES(?,?,?)",
        (account, message_id, folder),
    )
    conn.commit()


def update_sync_status(conn, account: str):
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO sync_status(account, last_synced) VALUES(?, ?)",
        (account, datetime.utcnow().isoformat()),
    )
    conn.commit()


def safe_filename(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]+", "", s)
    return s[:80] if s else "no_subject"


def decode_mime_words(s: str) -> str:
    if not s:
        return ""
    parts = decode_header(s)
    decoded = []
    for part, enc in parts:
        if isinstance(part, bytes):
            decoded.append(part.decode(enc or "utf-8", errors="replace"))
        else:
            decoded.append(part)
    return "".join(decoded)


def extract_text(msg: email.message.EmailMessage) -> str:
    # Priorité au text/plain, sinon fallback text/html (brut)
    if msg.is_multipart():
        for p in msg.walk():
            ctype = p.get_content_type()
            disp = str(p.get("Content-Disposition", "")).lower()
            if ctype == "text/plain" and "attachment" not in disp:
                try:
                    return p.get_content()
                except Exception:
                    payload = p.get_payload(decode=True)
                    if payload:
                        return payload.decode("utf-8", errors="replace")
        # fallback
        for p in msg.walk():
            ctype = p.get_content_type()
            disp = str(p.get("Content-Disposition", "")).lower()
            if ctype == "text/html" and "attachment" not in disp:
                try:
                    return p.get_content()
                except Exception:
                    payload = p.get_payload(decode=True)
                    if payload:
                        return payload.decode("utf-8", errors="replace")
        return ""
    else:
        try:
            return msg.get_content()
        except Exception:
            payload = msg.get_payload(decode=True)
            if payload:
                return payload.decode("utf-8", errors="replace")
            return ""


def format_since_date(dt: datetime) -> str:
    # IMAP "SINCE" attend un format du type 03-Jan-2026 (mois en anglais)
    return dt.strftime("%d-%b-%Y")


def main():
    ensure_dirs()
    conn = init_db()

    account = input("Adresse email (ex: prenom.nom@imtbs-tsp.eu): ").strip()
    password = getpass("Mot de passe IMAP (saisie cachée): ")

    # Date par défaut = 30 jours
    default_since = datetime.now() - timedelta(days=30)
    since_raw = input(
        f"Télécharger les emails depuis (YYYY-MM-DD) [défaut: {default_since.date()}]: "
    ).strip()

    if since_raw:
        since_dt = datetime.strptime(since_raw, "%Y-%m-%d")
    else:
        since_dt = default_since

    since_imap = format_since_date(since_dt)
    folder = "INBOX"

    print(f"[INFO] Connexion IMAP: {HOST}:{PORT} ...")
    imap = imaplib.IMAP4_SSL(HOST, PORT)
    imap.login(account, password)
    imap.select(folder)

    print(f"[INFO] Recherche IMAP depuis {since_imap} ...")
    # On récupère les messages SINCE la date (incluse)
    status, data = imap.search(None, f'(SINCE "{since_imap}")')
    if status != "OK":
        raise RuntimeError("IMAP search a échoué.")

    msg_ids = data[0].split()
    print(f"[INFO] {len(msg_ids)} emails trouvés (après filtrage date).")

    downloaded = 0
    skipped = 0

    for mid in msg_ids:
        status, msg_data = imap.fetch(mid, "(RFC822)")
        if status != "OK" or not msg_data or not msg_data[0]:
            continue

        raw = msg_data[0][1]
        msg = email.message_from_bytes(raw, policy=policy.default)

        message_id = (msg.get("Message-ID") or "").strip()
        if not message_id:
            # fallback si Message-ID absent
            message_id = f"NO_MESSAGE_ID_{mid.decode('utf-8', errors='ignore')}"

        if was_downloaded(conn, account, message_id):
            skipped += 1
            continue

        subject = decode_mime_words(msg.get("Subject", ""))
        sender = decode_mime_words(msg.get("From", ""))
        date = decode_mime_words(msg.get("Date", ""))

        body = extract_text(msg).strip()

        # Nom de fichier: date + sujet + hash court sur message_id
        short_id = abs(hash(message_id)) % (10**10)
        fname = f"{since_dt.strftime('%Y%m')}_{safe_filename(subject)}_{short_id}.md"
        path = os.path.join(EMAIL_DIR, fname)

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {subject}\n\n")
            f.write(f"**From:** {sender}\n\n")
            f.write(f"**Date:** {date}\n\n")
            f.write(f"**Message-ID:** {message_id}\n\n")
            f.write("---\n\n")
            f.write(body + "\n")

        mark_downloaded(conn, account, message_id, folder)
        downloaded += 1

    update_sync_status(conn, account)
    imap.logout()

    print(f"[DONE] Emails sauvegardés: {downloaded}")
    print(f"[DONE] Emails ignorés (déjà présents): {skipped}")
    print(f"[DONE] Dossier: {EMAIL_DIR}")
    print(f"[DONE] Cache SQLite: {DB_PATH}")


if __name__ == "__main__":
    main()
