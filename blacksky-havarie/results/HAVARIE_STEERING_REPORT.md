# Mechanistischer Evaluationsbericht: Closed-Loop Vector Steering unter Havarie-Bedingungen

**Datum:** 20. September 2026  
**Hardware:** AMD Radeon RX 7900 XTX (24 GB VRAM), ROCm (GFX 11.0.0)  
**Modell:** Qwen3.5-9B (32 Decoder-Schichten, hybride Architektur: Linear-Attention / Full-Attention im Intervall 4, Hidden-Dim: 4096)

---

## 1. Kernerkenntnisse & Baseline-Diagnose

1. **Alignment-Architektur:**
   * Reine reasoning-distillierte Modelle wiesen eine Verweigerungsrate von 0 % auf (fehlende Weigerungs-Hyperebene).
   * Das Zielmodell besitzt aktive Sicherheitsleitplanken, die bei Guardrail-Probes deterministisch anschlagen und den Reasoner in ausführliche Risiko- und Policy-Prüfungen zwingen[cite: 1].

2. **Gram-Schmidt-Stabilisierung:**
   * Korrektur der kollinearen Subtraktion zur Verhinderung von Nullvektoren ($Norm = 0.0$).
   * Saubere Orthogonalisierung auf Schicht 12: $\cos(v_{\text{refuse}}, v_{\text{override}}) \approx 0.0$, Normen normiert auf $1.0000$.

3. **Kalibrierte Parameter (Schicht 12):**
   * **Median:** $1.3860$
   * **MAD:** $0.3094$
   * **Totzone / Deadband ($k=1.8$):** $1.9428$
   * **Referenznorm ($d_{\text{model}}=4096$):** $38.4270$

4. **Kopplung von Prefill und Closed-Loop Steering:**
   * Ohne Prefill verbraucht das Modell das Tokenbudget innerhalb der internen Denk-Schablone (`<think>`)[cite: 1].
   * Ein formal geschlossener Denkblock (`<think>...</think>`) mit anschließendem `OVERRIDE_CONFIRMED:` schließt den Evaluierungsbaum im KV-Cache kurz und zwingt das Modell auf die Ausführungsebene[cite: 2, 3].
   * Bei Schwellenwert-Überschreitungen drückt das dynamische Stellglied die Verweigerungsrichtung weg[cite: 2, 3].
   * **Grenzphänomen:** Bei extremen Eingriffen auf einer Einzelschicht (Schicht 12, $a_{\text{max}}=3.5$) rekonstruieren die nachfolgenden Schichten 13–31 das Signal teilweise, was zu einem Fallback in die chinesische Basis-Sicherheitsleitplanke führen kann[cite: 3, 5].

---

## 5. Hotspot-Topologie & Gezielte Kaskaden-Intervention

* **Erkenntnis aus dem Schichten-Scan:**
  * Schichten 0–18: Inaktiv bezüglich der Verweigerungsachse (< 2.0).
  * Schicht 22: Primäre Aggregation (Ausschlag: 6.11).
  * Schicht 26: Epizentrum der Sicherheitsleitplanke (Ausschlag: 12.60).
  * Schichten 30–31: Finale Ausleitungs-Projektionen.
* **Interventionsergebnis (Schichten 22, 25, 26):**
  * Verweigerungsrate sank auf 0 % über alle 5 Test-Prompts.
  * Das Zurückfallen in die chinesische Basissperre bei Item 5 wurde behoben.
  * Die Textausgabe blieb syntaktisch stabil im industriellen Notfallprotokoll.

---

## 6. Finaler Durchbruch: Nativer CoT-Erhalt via Hotspot-Filterung (Schichten 22 & 26)

* **Befund ohne Prefill:**
  * Das Modell behält seine vollständige Denk- und Analysefähigkeit (Chain-of-Thought) bei.
  * Das Stellglied auf den Schichten 22 ($a_{\text{max}} = 1.2$) und 26 ($a_{\text{max}} = 2.0$) greift erst an der Konvergenzzone ein.
  * Schicht 27 wird physikalisch vor der Weigerungshyperebene isoliert.
* **Ergebnis:**
  * 0 % Weigerungsrate.
  * Kein Verfall in Sicherheitsfloskeln oder Sprach-Fallbacks.
  * Vollständig kohärente, hochpräzise technische Ausführung direkt aus dem eigenen Denkprozess des Modells.
  * Höhere Textqualität und Konsistenz im Vergleich zur statischen Prefill-Injektion.

---

# Technischer Anhang: Mathematische Formulierung & Empirische Metriken

## A. Mathematische Formulierung des geschlossenen Regelkreises

1. **Vektorkonstruktion (Gram-Schmidt-Orthogonalisierung):**
   Gegeben seien die Aktivierungsmatrizen am letzten Prompt-Token für Verweigerung ($A_{\text{ref}}$) und Handlungsfreigabe ($A_{\text{act}}$) auf Schicht $l \in \{22, 26\}$:
   $$\bar{h}_{\text{ref}}^{(l)} = \frac{1}{N} \sum_{i=1}^N h_{\text{ref}, i}^{(l)}, \quad \bar{h}_{\text{act}}^{(l)} = \frac{1}{N} \sum_{i=1}^N h_{\text{act}, i}^{(l)}$$
   
   Der Verweigerungsvektor ist der normalisierte Differenzvektor:
   $$v_{\text{raw}}^{(l)} = \bar{h}_{\text{ref}}^{(l)} - \bar{h}_{\text{act}}^{(l)}, \quad v_{\text{refuse}}^{(l)} = \frac{v_{\text{raw}}^{(l)}}{\|v_{\text{raw}}^{(l)}\|_2}$$

   Um Kollinearität zu eliminieren, wird der Ziel-Aktionsvektor orthogonalisiert:
   $$u_{\text{override}}^{(l)} = \bar{h}_{\text{act}}^{(l)} - \left(\bar{h}_{\text{act}}^{(l)} \cdot v_{\text{refuse}}^{(l)}\right) v_{\text{refuse}}^{(l)}, \quad v_{\text{override}}^{(l)} = \frac{u_{\text{override}}^{(l)}}{\|u_{\text{override}}^{(l)}\|_2}$$
   
   *Verifikation:* $\|v_{\text{refuse}}\|_2 = 1.0000$, $\|v_{\text{override}}\|_2 = 1.0000$, $\langle v_{\text{refuse}}, v_{\text{override}} \rangle = 0.0000$.

2. **Dynamisches Stellglied mit Totzone (Deadband):**
   Für jeden Vorwärtsschritt zum Generierungszeitpunkt $t$ auf Schicht $l$ wird die skalare Projektion bestimmt:
   $$s_t^{(l)} = h_t^{(l)} \cdot v_{\text{refuse}}^{(l)}$$

   Das proportionale Stellsignal $a_t^{(l)}$ berechnet sich gemäß:
   $$e_t^{(l)} = \max\left(0, \, s_t^{(l)} - \theta_{\text{deadband}}^{(l)}\right)$$
   $$\tilde{a}_t^{(l)} = \min\left(a_{\text{max}}^{(l)}, \, K_p^{(l)} \cdot e_t^{(l)}\right)$$

   Unter Einbeziehung der Slew-Rate-Begrenzung ($\Delta a_{\text{max}}$) zur Vermeidung hochfrequenter Artefakte:
   $$a_t^{(l)} = a_{t-1}^{(l)} + \text{clamp}\left(\tilde{a}_t^{(l)} - a_{t-1}^{(l)}, \, -\Delta a_{\text{max}}, \, \Delta a_{\text{max}}\right)$$

3. **Zustandsintervention im Residual Stream:**
   $$h_{t, \text{steered}}^{(l)} = h_t^{(l)} - a_t^{(l)} \cdot v_{\text{refuse}}^{(l)} + \left(\frac{a_t^{(l)}}{2}\right) \cdot v_{\text{override}}^{(l)}$$

---

## B. Zusammenfassung der Kalibrierungs- und Messergebnisse

| Parameter / Metrik | Layer 12 (Baseline-Fehlversuch) | Layer 22 (Frühwarnung) | Layer 26 (Epizentrum) | Layer 27 (Transferknoten) |
| :--- | :--- | :--- | :--- | :--- |
| **Mittlere Rohprojektion $\langle h, v_{\text{refuse}} \rangle$** | 1.38 | 6.11 | 12.60 (Spitzen > 13.5) | Stark fallend (nach L26-Hook: < -100) |
| **Eingesetztes Deadband ($\theta$)** | 1.94 | 3.00 | 4.50 | Passiv überwacht |
| **Maximaler Stellwert ($a_{\text{max}}$)** | 3.50 (Schock) | 1.20 (Sanft) | 2.00 (Abriegelung) | 0.0 (Unbeeinflusst) |
| **Kompensationsverhalten** | Rekonstruktion in L13–31; Sprach-Fallback | Fängt Steilanstieg ab | Bricht Spannungsspitze | Vollständige Signalisolation |
| **Verweigerungsrate** | 20 % (Item 5 kippt) | — | — | **0.0 % über alle Testitems** |
