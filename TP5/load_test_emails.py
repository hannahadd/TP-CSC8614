# TP5/load_test_emails.py
import os
import re
from typing import Dict, List

EMAIL_DIR = os.path.join("TP5", "data", "test_emails")

RE_BODY = re.compile(r"CORPS:\s*<<<\s*(.*?)\s*>>>", re.DOTALL)
RE_ID = re.compile(r"email_id:\s*(\S+)")
RE_SUBJECT = re.compile(r"subject:\s*\"(.*)\"")
RE_FROM = re.compile(r"from:\s*\"(.*)\"")


def load_one_email(path: str) -> Dict[str, str]:
    txt = open(path, "r", encoding="utf-8").read()

    email_id_match = RE_ID.search(txt)
    subject_match = RE_SUBJECT.search(txt)
    from_match = RE_FROM.search(txt)
    body_match = RE_BODY.search(txt)

    email_id = email_id_match.group(1).strip() if email_id_match else os.path.splitext(os.path.basename(path))[0]
    subject = subject_match.group(1).strip() if subject_match else ""
    from_ = from_match.group(1).strip() if from_match else ""
    body = body_match.group(1).strip() if body_match else ""

    return {
        "email_id": email_id,
        "subject": subject,
        "from": from_,
        "body": body,
        "path": path,
    }


def load_all_emails() -> List[Dict[str, str]]:
    files = []
    for fn in os.listdir(EMAIL_DIR):
        if fn.endswith(".md") or fn.endswith(".txt"):
            files.append(os.path.join(EMAIL_DIR, fn))

    def _sort_key(p: str):
        base = os.path.basename(p)
        m = re.search(r"E(\d+)", base, flags=re.IGNORECASE)
        return (int(m.group(1)) if m else 10**9, base)

    files = sorted(files, key=_sort_key)

    emails = [load_one_email(p) for p in files]
    return emails


if __name__ == "__main__":
    emails = load_all_emails()
    print(f"Loaded {len(emails)} emails")
    for e in emails:
        print(f"- {e['email_id']}: {e['subject']} ({os.path.basename(e['path'])})")
