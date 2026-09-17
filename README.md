# Sovereign LLM Vault Appliance

Autarke, air-gapped On-Premise Dokumenten-Audit Appliance für Berufsgeheimnisträger (§ 203 StGB, NIS2, DSGVO).

## Key Facts & Performance
- **Inferenz-Engine:** Native `llama-server` Kompilierung mit FlashAttention-2 & ROCm-Hardwarebeschleunigung.
- **Hardware-Target:** AMD Radeon RX 7900 XTX (24 GB VRAM) auf Bare-Metal Linux.
- **Modell:** Qwen 2.5 Coder 14B Instruct (GGUF Q4_K_M).
- **Prompt-Ingestion-Rate:** > 1.400 Tokens/Sekunde.
- **Generierungs-Geschwindigkeit:** ~50 Tokens/Sekunde.
- **Netzwerk-Isolation:** Striktes Loopback (`127.0.0.1:8080`), 0 Byte Cloud-Abfluss, voll funktionsfähig ohne Internetverbindung.

---

## Live-Benchmarks & Nachweise

### Fall 1: DSGVO, NIS2 & Quellcode-Leck (Stresstest mit OCR-Fehlern)
- **Prompt-Ingestion:** 1.421,2 T/s | **Generierung:** 51,9 T/s | **Laufzeit:** 11,96 s
- **Ergebnis:** Behördliche 72h-Notfrist isoliert, US-Haftungsbegrenzung ($500) und unzulässiges Modell-Training als Kritisch bewertet.

<p align="center">
<img src="media/benchmark_nis2_stresstest.png" alt="Stresstest NIS2 Benchmark" width="95%">
</p>

### Fall 2: Steuerforensik & AfA-Audit (§ 6 EStG / vGA)
- **Prompt-Ingestion:** 1.369,5 T/s | **Generierung:** 50,1 T/s | **Laufzeit:** 23,87 s
- **Ergebnis:** Anschaffungsnahe Herstellungskosten (§ 6 Abs. 1 Nr. 1a EStG) erkannt, Zahlungsflüsse extrahiert, Fristenkollision getrennt.

<p align="center">
<img src="media/benchmark_tax_forensics.png" alt="Steuerforensik Benchmark" width="95%">
</p>

---

## Video-Walkthrough
Die vollständige Terminal-Aufzeichnung der Inferenz und der Systemauslastung (`nvtop` / `systemd`) ist im Repository hinterlegt:

▶️ **[Video ansehen / herunterladen: `media/benchmark_proof_walkthrough.webm`](media/benchmark_proof_walkthrough.webm)**

---

## Architektur
text
[ Netzwerklaufwerk: /srv/vault/inbox/ ]
│
▼ (Atomarer File-Watchdog)
[ vault_watchdog.py (Injects Governance & Audit Prompts) ]
│
▼ (Loopback REST POST /v1/chat/completions)
[ sovereign-llm.service (Bare-Metal llama-server @ RX 7900 XTX) ]
│
▼ (Deterministische Extraktion)
[ /srv/vault/outbox/xyz_audit.json ] & [ /srv/vault/archive/ ]
## Repository-Struktur
text
.
├── media/               # Benchmark-Screenshots und Videoaufzeichnung
├── systemd/             # Init-Units für llama-server und Watchdog-Daemon
├── scripts/             # Gehärteter Watchdog mit atomarer Dateiverarbeitung
├── benchmarks/          # Synthetische Testakten und JSON-Audit-Outputs
└── README.md
## Lizenz
MIT License. Frei verwendbar für Managed Service Provider (MSPs) und IT-Systemhäuser.
