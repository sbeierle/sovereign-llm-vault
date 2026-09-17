#!/usr/bin/env python3
import time
import json
import shutil
import urllib.request
from pathlib import Path

INBOX = Path("/srv/vault/inbox")
OUTBOX = Path("/srv/vault/outbox")
ARCHIVE = Path("/srv/vault/archive")
API_URL = "http://127.0.0.1:8080/v1/chat/completions"

SYSTEM_PROMPT = """Du bist ein lokaler forensischer Dokumenten-Auditor fuer Berufsgeheimnistraeger (§ 203 StGB, NIS2, DSGVO).
Analysiere den uebergebenen Textauszug praezise und extrahiere:
1. "dokumenten_typ": (z. B. Vertrag, Schriftsatz, Mandantenkorrespondenz)
2. "risiko_analyse":
   - "risiko_level": ("Niedrig", "Mittel", "Kritisch")
   - "schadenspotenzial": Kurze Erlaeuterung
   - "normen_konflikte": Relevante Gesetze (§ 203 StGB, DSGVO Art. 9, etc.)
3. "extraktions_tabelle": Wichtigste Fristen, Parteien, Geldbetraege oder Haftungsklauseln als Key-Value
4. "handlungsempfehlung": Konkrete Sofortmassnahmen fuer den Sachbearbeiter

Antworte ausnahmslos in validem JSON. Verwende keine Markdown-Formatierung um das JSON."""

def process_file(file_path: Path):
    if not file_path.exists():
        return

    print(f"[*] Verarbeite: {file_path.name}")
    start_t = time.perf_counter()
    
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[!] Lesefehler: {e}")
        return

    payload = {
        "model": "qwen2.5-coder-14b",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Dokumentinhalt:\n{content[:25000]}"}
        ],
        "temperature": 0.1
    }

    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            result_text = data["choices"][0]["message"]["content"].strip()
            timings = data.get("timings", {})
    except Exception as e:
        print(f"[!] Inferenz-Fehler: {e}")
        return

    elapsed = time.perf_counter() - start_t
    
    if result_text.startswith("```"):
        result_text = result_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    out_file = OUTBOX / f"{file_path.stem}_audit.json"
    out_file.write_text(result_text, encoding="utf-8")

    ts = int(time.time())
    archive_target = ARCHIVE / f"{file_path.stem}_{ts}{file_path.suffix}"
    try:
        if file_path.exists():
            shutil.move(str(file_path), str(archive_target))
    except Exception as e:
        print(f"[!] Archivierungsfehler: {e}")

    prompt_tps = timings.get("prompt_per_second", 0)
    eval_tps = timings.get("predicted_per_second", 0)
    print(f"[+] Fertig in {elapsed:.2f}s | Prompt: {prompt_tps:.1f} T/s | Eval: {eval_tps:.1f} T/s")
    print(f"[+] Ergebnis gesichert: {out_file}\n")

def main():
    print("[*] Sovereign Vault Watchdog aktiv. Ueberwache /srv/vault/inbox/ ...")
    while True:
        try:
            for f in list(INBOX.glob("*")):
                if f.is_file() and not f.name.startswith(".") and not f.name.endswith(".tmp"):
                    size_1 = f.stat().st_size
                    time.sleep(0.2)
                    if not f.exists():
                        continue
                    size_2 = f.stat().st_size
                    if size_1 == size_2 and size_1 > 0:
                        process_file(f)
        except Exception as e:
            print(f"[!] Loop-Fehler: {e}")
        time.sleep(0.5)

if __name__ == "__main__":
    main()
