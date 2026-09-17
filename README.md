# Sovereign LLM Vault Appliance

Autarke, air-gapped On-Premise Dokumenten-Audit Appliance für Berufsgeheimnisträger (§ 203 StGB, NIS2, DSGVO).

## Key Facts & Performance
- **Inferenz-Engine:** Native `llama-server` Kompilierung mit FlashAttention-2 & ROCm-Hardwarebeschleunigung.
- **Hardware-Target:** AMD Radeon RX 7900 XTX (24 GB VRAM) auf Bare-Metal Linux.
- **Modell:** Qwen 2.5 Coder 14B Instruct (GGUF Q4_K_M).
- **Prompt-Ingestion-Rate:** > 1.400 Tokens/Sekunde.
- **Generierungs-Geschwindigkeit:** ~50 Tokens/Sekunde.
- **Netzwerk-Isolation:** Striktes Loopback (`127.0.0.1:8080`), 0 Byte Cloud-Abfluss, voll funktionsfähig ohne Internetverbindung.

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
- `systemd/`: Init-Units für LLM-Server und Vault-Watchdog-Daemon.
- `scripts/`: Gehärteter Watchdog mit atomarer Dateiverarbeitung.
- `benchmarks/`: Synthetische Testakten (Datenschutz/NIS2, Steuer/AfA/vGA) und zugehörige JSON-Audit-Outputs.

## Lizenz
MIT License. Frei verwendbar für Managed Service Provider (MSPs) und IT-Systemhäuser.

## Verifizierte Benchmarks (Live-Messungen)

### Fall 1: DSGVO, NIS2 & Quellcode-Leck (Stresstest mit OCR-Fehlern)
- **Prompt-Ingestion:** 1.421,2 T/s
- **Generierung:** 51,9 T/s
- **Laufzeit:** 11,96 s

![Fall 1 Benchmark](media/benchmark_nis2_stresstest.png)

### Fall 2: Steuerforensik & AfA-Audit (§ 6 EStG / vGA)
- **Prompt-Ingestion:** 1.369,5 T/s
- **Generierung:** 50,1 T/s
- **Laufzeit:** 23,87 s

![Fall 2 Benchmark](media/benchmark_tax_forensics.png)
