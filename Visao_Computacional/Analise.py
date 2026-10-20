import os
import cv2
import base64
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# =====================================================================
# 1. CONFIGURAÇÃO DE CAMINHOS
# =====================================================================
DIR_ATUAL = os.path.dirname(os.path.abspath(__file__))
DIR_RAIZ = os.path.dirname(DIR_ATUAL)

CAMINHO_MODELO = os.path.join(DIR_RAIZ, 'Machine_Learning', 'Registro', 'ficheiro.joblib')
DIR_ASSETS = os.path.join(DIR_ATUAL, 'assets')
DIR_TEMPLATES = os.path.join(DIR_ASSETS, 'templates')
DIR_TESTES = os.path.join(DIR_ASSETS, 'testes')
DIR_OUTPUT = os.path.join(DIR_ATUAL, 'output')

os.makedirs(DIR_OUTPUT, exist_ok=True)

# =====================================================================
# 2. CARREGAMENTO DO MODELO PREDITIVO (ETAPA 2)
# =====================================================================
modelo = None
scaler = None
features_esperadas = None

def carregar_modelo():
    global modelo, scaler, features_esperadas
    if not os.path.exists(CAMINHO_MODELO):
        return False
    try:
        artefato = joblib.load(CAMINHO_MODELO)
        if isinstance(artefato, dict):
            modelo = artefato.get('modelo') or artefato.get('model')
            scaler = artefato.get('scaler')
        else:
            modelo = artefato

        if hasattr(modelo, 'feature_names_in_'):
            features_esperadas = list(modelo.feature_names_in_)
        return True
    except Exception as e:
        print(f"[ERRO] Falha ao carregar ficheiro.joblib: {e}")
        return False

carregar_modelo()

# =====================================================================
# 3. MOTOR DE VISÃO COMPUTACIONAL INVARIANTE À ROTAÇÃO (OPENCV)
# =====================================================================

def ler_imagem_caminho(caminho):
    if not os.path.exists(caminho):
        return None
    bytes_img = np.fromfile(caminho, dtype=np.uint8)
    return cv2.imdecode(bytes_img, cv2.IMREAD_COLOR)

def ler_imagem_upload(arquivo_storage):
    bytes_img = np.frombuffer(arquivo_storage.read(), np.uint8)
    return cv2.imdecode(bytes_img, cv2.IMREAD_COLOR)

def salvar_imagem_seguro(caminho, img):
    sucesso, buffer = cv2.imencode('.png', img)
    if sucesso:
        buffer.tofile(caminho)

def imagem_para_base64(img):
    _, buffer = cv2.imencode('.png', img)
    return "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')


def extrair_caracteristicas_peca(img_bgr):
    """
    Isola a peça principal e calcula métricas 100% INVARIANTES À ROTAÇÃO (45°, 90°, etc.),
    evitando que o bounding box quadrado diminua a escala da peça quando as orelhas giram.
    """
    h_orig, w_orig = img_bgr.shape[:2]
    escala = 500.0 / max(h_orig, w_orig)
    img_prop = cv2.resize(img_bgr, (int(w_orig * escala), int(h_orig * escala)), interpolation=cv2.INTER_AREA)

    # Canvas amplo com fundo branco
    canvas = np.full((600, 600, 3), 255, dtype=np.uint8)
    y_off = (600 - img_prop.shape[0]) // 2
    x_off = (600 - img_prop.shape[1]) // 2
    canvas[y_off:y_off+img_prop.shape[0], x_off:x_off+img_prop.shape[1]] = img_prop

    cinza = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(cinza, (7, 7), 0)

    # Segmentação por saturação HSV (ignora barras cinzas no rodapé) + fallback Otsu
    hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
    _, mask_cor = cv2.threshold(hsv[:, :, 1], 35, 255, cv2.THRESH_BINARY)
    _, mask_otsu = cv2.threshold(blur, 220, 255, cv2.THRESH_BINARY_INV)
    mascara = mask_cor if np.sum(mask_cor > 0) > 15000 else mask_otsu

    kernel = np.ones((5, 5), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=1)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=1)

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None

    contorno_ext = max(contornos, key=cv2.contourArea)

    # NORMALIZAÇÃO POR CÍRCULO MÍNIMO (Invariante à rotação de 45°!)
    # Diferente do boundingRect quadrado, o círculo circunscrito nunca muda de raio quando a peça gira!
    (cx, cy), raio = cv2.minEnclosingCircle(contorno_ext)
    if raio < 10:
        return None

    # Redimensiona o canvas inteiro para que o raio externo da peça seja sempre exatamente 180 pixels
    fator_escala = 180.0 / raio
    M = cv2.getRotationMatrix2D((cx, cy), 0, fator_escala)
    # Centraliza a peça exatamente no meio de uma janela 420x420
    M[0, 2] += (210.0 - cx)
    M[1, 2] += (210.0 - cy)

    peca_bgr = cv2.warpAffine(canvas, M, (420, 420), flags=cv2.INTER_AREA, borderValue=(255, 255, 255))
    peca_cinza = cv2.cvtColor(peca_bgr, cv2.COLOR_BGR2GRAY)
    peca_blur = cv2.GaussianBlur(peca_cinza, (7, 7), 1.5)

    # Isola novamente com o canal HSV/Cinza na escala perfeitamente padronizada
    peca_hsv = cv2.cvtColor(peca_bgr, cv2.COLOR_BGR2HSV)
    _, mask_peca_cor = cv2.threshold(peca_hsv[:, :, 1], 35, 255, cv2.THRESH_BINARY)
    _, mask_peca_otsu = cv2.threshold(peca_blur, 215, 255, cv2.THRESH_BINARY_INV)
    mask_peca = mask_peca_cor if np.sum(mask_peca_cor > 0) > 10000 else mask_peca_otsu
    mask_peca = cv2.morphologyEx(mask_peca, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    # Filtro Canny Edge Detection (exigido no PDF)
    bordas_canny = cv2.Canny(peca_blur, 45, 135)
    bordas_canny = cv2.bitwise_and(bordas_canny, bordas_canny, mask=cv2.dilate(mask_peca, kernel, iterations=1))

    # Contornos e Hierarquia de Furos
    contornos_peca, hierarquia = cv2.findContours(mask_peca, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos_peca:
        return None

    idx_maior = int(np.argmax([cv2.contourArea(c) for c in contornos_peca]))
    contorno_principal = contornos_peca[idx_maior]

    area_contorno = float(cv2.contourArea(contorno_principal))
    hull = cv2.convexHull(contorno_principal)
    area_hull = float(cv2.contourArea(hull))

    # Solidez e Razão de Área em relação ao Fecho Convexo (100% invariantes à rotação!)
    solidez = (area_contorno / (area_hull + 1e-5)) * 100.0

    furos_validos = []
    area_furo_central = 0.0
    if hierarquia is not None:
        for i, c in enumerate(contornos_peca):
            if hierarquia[0][i][3] == idx_maior:
                a_furo = cv2.contourArea(c)
                if a_furo > 35:
                    furos_validos.append(c)
                    if a_furo > area_furo_central:
                        area_furo_central = a_furo

    # Proporção da área líquida e do furo central em relação à área da peça (invariante à rotação)
    pixels_peca = float(np.sum(mask_peca > 0))
    area_relativa = (pixels_peca / (area_hull + 1e-5)) * 100.0
    furo_central_relativo = (area_furo_central / (area_contorno + 1e-5)) * 100.0
    densidade_canny = float(np.sum(bordas_canny > 0)) / (pixels_peca + 1e-5) * 100.0

    return {
        "img_peca": peca_bgr,
        "canny": bordas_canny,
        "contorno_principal": contorno_principal,
        "furos_contornos": furos_validos,
        "area_relativa": area_relativa,
        "solidez": solidez,
        "furo_central_relativo": furo_central_relativo,
        "densidade_canny": densidade_canny,
        "num_furos": len(furos_validos)
    }


def comparar_pecas(img_ref, img_teste, nome_saida="resultado_inspecao.png",
                   tol_area=3.5, tol_canny=20.0, tol_hu=0.075):
    """
    Compara a peça de teste com a referência de forma invariante à rotação.
    """
    feat_ref = extrair_caracteristicas_peca(img_ref)
    feat_teste = extrair_caracteristicas_peca(img_teste)

    if feat_ref is None or feat_teste is None:
        return {"erro": "Não foi possível isolar os contornos de uma das imagens."}

    # 1. Variação de Área Estrutural (combina área líquida normalizada pelo raio + solidez do contorno)
    diff_area_rel = abs(feat_teste["area_relativa"] - feat_ref["area_relativa"]) / (feat_ref["area_relativa"] + 1e-5) * 100.0
    diff_solidez = abs(feat_teste["solidez"] - feat_ref["solidez"]) / (feat_ref["solidez"] + 1e-5) * 100.0
    diff_area_pct = max(diff_area_rel, diff_solidez)

    # 2. Variação proporcional do Furo Central (detecta obstrução sem sofrer com rotação de 45°)
    diff_furo_central_pct = abs(feat_teste["furo_central_relativo"] - feat_ref["furo_central_relativo"]) / (feat_ref["furo_central_relativo"] + 1e-5) * 100.0

    # 3. Variação de Bordas Canny (detecta trincas, fissuras, ferrugem)
    diff_canny_pct = abs(feat_teste["densidade_canny"] - feat_ref["densidade_canny"]) / (feat_ref["densidade_canny"] + 1e-5) * 100.0

    # 4. Distância Geométrica Invariante à Rotação (Hu Moments)
    dist_forma_hu = float(cv2.matchShapes(feat_ref["contorno_principal"], feat_teste["contorno_principal"], cv2.CONTOURS_MATCH_I1, 0.0))

    # Regras de Decisão com base nos Sliders
    motivos_reprovacao = []

    if diff_area_pct > tol_area:
        motivos_reprovacao.append(f"Variação na massa/área da peça ({diff_area_pct:.2f}% > limite de {tol_area:.1f}%) — indica quebra de orelha ou excesso de material.")

    if diff_furo_central_pct > (tol_area * 3.0):
        motivos_reprovacao.append(f"Obstrução ou deformação crítica no furo central (variação de {diff_furo_central_pct:.1f}% na cavidade interna).")

    if diff_canny_pct > tol_canny:
        motivos_reprovacao.append(f"Anomalia superficial no filtro Canny ({diff_canny_pct:.2f}% > limite de {tol_canny:.1f}%) — indica fissura, trinca ou rebarba.")

    if dist_forma_hu > tol_hu:
        motivos_reprovacao.append(f"Geometria externa fora do gabarito (Índice Hu: {dist_forma_hu:.4f} > limite de {tol_hu:.3f}).")

    aprovada = len(motivos_reprovacao) == 0
    status_str = "APROVADA" if aprovada else "REPROVADA"
    cor_status = (0, 210, 0) if aprovada else (0, 0, 255)

    # Desenho do Diagnóstico Visual
    vis_contornos = feat_teste["img_peca"].copy()
    cv2.drawContours(vis_contornos, [feat_teste["contorno_principal"]], -1, cor_status, 3)
    cv2.drawContours(vis_contornos, feat_teste["furos_contornos"], -1, (255, 180, 0), 2)

    cv2.rectangle(vis_contornos, (0, 0), (420, 68), (20, 20, 20), -1)
    cv2.putText(vis_contornos, f"Status: {status_str}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor_status, 2)
    cv2.putText(vis_contornos, f"Area: {diff_area_pct:.1f}% | Canny: {diff_canny_pct:.1f}% | Hu: {dist_forma_hu:.3f}",
                (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)

    canny_bgr = cv2.cvtColor(feat_teste["canny"], cv2.COLOR_GRAY2BGR)
    cv2.rectangle(canny_bgr, (0, 0), (420, 45), (20, 20, 20), -1)
    cv2.putText(canny_bgr, "Filtro Canny Edge Detection", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    painel_final = np.hstack((vis_contornos, canny_bgr))
    caminho_salvo = os.path.join(DIR_OUTPUT, nome_saida)
    salvar_imagem_seguro(caminho_salvo, painel_final)

    return {
        "status_qualidade": status_str,
        "peca_aprovada": aprovada,
        "motivos_reprovacao": motivos_reprovacao if not aprovada else ["Peça íntegra! Geometria e superfície dentro das tolerâncias (invariante à rotação)."],
        "metricas_comparativas": {
            "variacao_area_percentual": round(diff_area_pct, 2),
            "variacao_furo_central_percentual": round(diff_furo_central_pct, 2),
            "variacao_bordas_canny_percentual": round(diff_canny_pct, 2),
            "distancia_geometrica_hu": round(dist_forma_hu, 4),
            "furos_esperados": feat_ref["num_furos"],
            "furos_detectados": feat_teste["num_furos"]
        },
        "imagem_diagnostico_salva_em": caminho_salvo,
        "imagem_base64": imagem_para_base64(painel_final)
    }


# =====================================================================
# 4. INTERFACE WEB (HTML + CSS + JS EMBUTIDO)
# =====================================================================
HTML_DASHBOARD = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>MetalTech S.A. - Inspeção Visual & Preditiva</title>
    <style>
        :root {
            --bg: #0f172a; --card: #1e293b; --border: #334155;
            --text: #f8fafc; --muted: #94a3b8; --accent: #38bdf8;
            --ok: #22c55e; --err: #ef4444;
        }
        * { box-sizing: border-box; font-family: 'Segoe UI', system-ui, sans-serif; }
        body { margin: 0; background: var(--bg); color: var(--text); padding: 24px; }
        header {
            display: flex; justify-content: space-between; align-items: center;
            border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 24px;
        }
        header h1 { margin: 0; font-size: 1.5rem; color: var(--accent); }
        .grid { display: grid; grid-template-columns: 380px 1fr; gap: 24px; }
        .card {
            background: var(--card); border: 1px solid var(--border);
            border-radius: 12px; padding: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        }
        h2 { margin-top: 0; font-size: 1.1rem; border-bottom: 1px solid var(--border); padding-bottom: 10px; }
        label { display: block; font-size: 0.85rem; color: var(--muted); margin: 12px 0 4px; }
        select, input[type="file"], input[type="number"] {
            width: 100%; padding: 9px; border-radius: 8px; border: 1px solid var(--border);
            background: #0f172a; color: var(--text); font-size: 0.9rem;
        }
        .slider-group { margin: 14px 0; }
        .slider-header { display: flex; justify-content: space-between; font-size: 0.85rem; }
        input[type="range"] { width: 100%; accent-color: var(--accent); cursor: pointer; margin-top: 6px; }
        button {
            width: 100%; padding: 12px; margin-top: 16px; border: none; border-radius: 8px;
            background: var(--accent); color: #0f172a; font-weight: 700; font-size: 0.95rem;
            cursor: pointer; transition: 0.2s;
        }
        button:hover { filter: brightness(1.1); }
        .badge {
            display: inline-block; padding: 6px 16px; border-radius: 999px;
            font-weight: 700; font-size: 0.95rem;
        }
        .badge.ok { background: rgba(34,197,94,0.2); color: var(--ok); border: 1px solid var(--ok); }
        .badge.err { background: rgba(239,68,68,0.2); color: var(--err); border: 1px solid var(--err); }
        .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 16px 0; }
        .metric-box { background: #0f172a; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid var(--border); }
        .metric-box span { display: block; font-size: 0.75rem; color: var(--muted); }
        .metric-box strong { font-size: 1.2rem; color: var(--accent); }
        .preview-img { width: 100%; border-radius: 8px; border: 1px solid var(--border); margin-top: 12px; }
        ul.motivos { padding-left: 20px; margin: 10px 0; color: #cbd5e1; font-size: 0.9rem; }
        .tabs { display: flex; gap: 10px; margin-bottom: 16px; }
        .tab-btn { flex: 1; padding: 8px; background: #0f172a; color: var(--muted); border: 1px solid var(--border); margin: 0; }
        .tab-btn.active { background: var(--accent); color: #0f172a; }
    </style>
</head>
<body>

<header>
    <div>
        <h1>⚙️ MetalTech Soluções Industriais S.A.</h1>
        <small style="color: var(--muted);">Sistema Integrado de Visão Computacional (OpenCV) & Manutenção Preditiva (IoT)</small>
    </div>
    <span class="badge ok">API Flask Online</span>
</header>

<div class="grid">
    <!-- COLUNA ESQUERDA: CONTROLES -->
    <div>
        <div class="card">
            <h2>1. Seleção de Peças (Visão Computacional)</h2>
            <div class="tabs">
                <button type="button" class="tab-btn active" onclick="mudarModo('pasta')">Arquivos do Projeto</button>
                <button type="button" class="tab-btn" onclick="mudarModo('upload')">Upload do PC</button>
            </div>

            <div id="modo-pasta">
                <label>Peça Boa (Referência / Gabarito):</label>
                <select id="sel_ref">
                    {% for arq in templates %}
                    <option value="{{ arq }}">{{ arq }}</option>
                    {% endfor %}
                </select>

                <label>Peça para Análise (Teste):</label>
                <select id="sel_teste">
                    {% for arq in testes %}
                    <option value="{{ arq }}">{{ arq }}</option>
                    {% endfor %}
                </select>
            </div>

            <div id="modo-upload" style="display:none;">
                <label>Enviar Imagem de Referência (Opcional):</label>
                <input type="file" id="file_ref" accept="image/*">
                <label>Enviar Imagem para Análise:</label>
                <input type="file" id="file_teste" accept="image/*">
            </div>

            <h2 style="margin-top:24px;">2. Sliders de Tolerância Industrial</h2>

            <div class="slider-group">
                <div class="slider-header">
                    <span>Tolerância de Área / Quebra:</span>
                    <strong id="val_area">3.5%</strong>
                </div>
                <input type="range" id="sl_area" min="0.5" max="20.0" step="0.5" value="3.5"
                       oninput="document.getElementById('val_area').innerText = this.value + '%'">
            </div>

            <div class="slider-group">
                <div class="slider-header">
                    <span>Tolerância Superficial Canny (Fissuras):</span>
                    <strong id="val_canny">20.0%</strong>
                </div>
                <input type="range" id="sl_canny" min="2.0" max="100.0" step="1.0" value="20.0"
                       oninput="document.getElementById('val_canny').innerText = this.value + '%'">
            </div>

            <div class="slider-group">
                <div class="slider-header">
                    <span>Tolerância Geométrica (Hu Moments):</span>
                    <strong id="val_hu">0.075</strong>
                </div>
                <input type="range" id="sl_hu" min="0.01" max="0.25" step="0.005" value="0.075"
                       oninput="document.getElementById('val_hu').innerText = parseFloat(this.value).toFixed(3)">
            </div>

            <button onclick="executarInspecao()">🔍 Analisar Peça Agora</button>
        </div>

        <!-- CARD PREDITIVO IOT -->
        <div class="card" style="margin-top: 20px;">
            <h2>3. Simulador de Sensores IoT (/predict)</h2>
            <label>Temperatura (°C):</label>
            <input type="number" id="iot_temp" value="75.0" step="0.5">
            <label>Vibração (mm/s):</label>
            <input type="number" id="iot_vib" value="2.5" step="0.1">
            <label>Pressão (bar):</label>
            <input type="number" id="iot_pres" value="95.0" step="1.0">
            <button style="background:#a855f7; color:white;" onclick="testarPredicao()">⚡ Prever Risco do Motor</button>
            <div id="res_iot" style="margin-top:12px; font-size:0.9rem;"></div>
        </div>
    </div>

    <!-- COLUNA DIREITA: RESULTADOS VISUAIS -->
    <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <h2 style="border:none; margin:0;">Laudo de Inspeção Visual Automatizada</h2>
            <span id="status_badge" class="badge">Aguardando análise...</span>
        </div>

        <div class="metrics">
            <div class="metric-box">
                <span>Variação de Área</span>
                <strong id="m_area">-</strong>
            </div>
            <div class="metric-box">
                <span>Variação Furo Central</span>
                <strong id="m_furo">-</strong>
            </div>
            <div class="metric-box">
                <span>Bordas Canny (Trincas)</span>
                <strong id="m_canny">-</strong>
            </div>
            <div class="metric-box">
                <span>Índice Geométrico Hu</span>
                <strong id="m_hu">-</strong>
            </div>
        </div>

        <div>
            <strong>Diagnóstico Técnico:</strong>
            <ul class="motivos" id="lista_motivos">
                <li>Selecione uma peça ao lado e clique em "Analisar Peça Agora".</li>
            </ul>
        </div>

        <img id="img_resultado" class="preview-img" style="display:none;" alt="Resultado OpenCV">
    </div>
</div>

<script>
    let modoAtual = 'pasta';

    function mudarModo(modo) {
        modoAtual = modo;
        document.getElementById('modo-pasta').style.display = modo === 'pasta' ? 'block' : 'none';
        document.getElementById('modo-upload').style.display = modo === 'upload' ? 'block' : 'none';
        const botoes = document.querySelectorAll('.tab-btn');
        botoes[0].classList.toggle('active', modo === 'pasta');
        botoes[1].classList.toggle('active', modo === 'upload');
    }

    async function executarInspecao() {
        const tol_area = document.getElementById('sl_area').value;
        const tol_canny = document.getElementById('sl_canny').value;
        const tol_hu = document.getElementById('sl_hu').value;

        let resposta;

        if (modoAtual === 'upload') {
            const fileTeste = document.getElementById('file_teste').files[0];
            const fileRef = document.getElementById('file_ref').files[0];
            if (!fileTeste) { alert("Selecione uma imagem de teste para enviar!"); return; }

            const formData = new FormData();
            formData.append('imagem_teste', fileTeste);
            if (fileRef) formData.append('imagem_referencia', fileRef);
            formData.append('tol_area', tol_area);
            formData.append('tol_canny', tol_canny);
            formData.append('tol_hu', tol_hu);

            resposta = await fetch('/inspect', { method: 'POST', body: formData });
        } else {
            const payload = {
                arquivo_referencia: document.getElementById('sel_ref').value,
                arquivo_teste: document.getElementById('sel_teste').value,
                tol_area: parseFloat(tol_area),
                tol_canny: parseFloat(tol_canny),
                tol_hu: parseFloat(tol_hu)
            };
            resposta = await fetch('/inspect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
        }

        const dados = await resposta.json();
        if (dados.erro) { alert(dados.erro); return; }

        const badge = document.getElementById('status_badge');
        badge.innerText = dados.status_qualidade;
        badge.className = 'badge ' + (dados.peca_aprovada ? 'ok' : 'err');

        document.getElementById('m_area').innerText = dados.metricas_comparativas.variacao_area_percentual + '%';
        document.getElementById('m_furo').innerText = dados.metricas_comparativas.variacao_furo_central_percentual + '%';
        document.getElementById('m_canny').innerText = dados.metricas_comparativas.variacao_bordas_canny_percentual + '%';
        document.getElementById('m_hu').innerText = dados.metricas_comparativas.distancia_geometrica_hu;

        const ul = document.getElementById('lista_motivos');
        ul.innerHTML = '';
        dados.motivos_reprovacao.forEach(m => {
            const li = document.createElement('li');
            li.innerText = m;
            ul.appendChild(li);
        });

        const img = document.getElementById('img_resultado');
        img.src = dados.imagem_base64;
        img.style.display = 'block';
    }

    async function testarPredicao() {
        const payload = {
            temperatura: parseFloat(document.getElementById('iot_temp').value),
            vibracao: parseFloat(document.getElementById('iot_vib').value),
            pressao: parseFloat(document.getElementById('iot_pres').value)
        };
        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const d = await res.json();
        if (d.erro) {
            document.getElementById('res_iot').innerHTML = `<strong style="color:#ef4444">${d.erro}</strong>`;
            return;
        }
        const cor = d["Risco de Falha"] === "Sim" ? "#ef4444" : "#22c55e";
        const det = d.motivo_alerta ? `<br><small style="color:#94a3b8">${d.motivo_alerta}</small>` : "";
        document.getElementById('res_iot').innerHTML =
            `<strong style="color:${cor}">${d.mensagem}</strong> (Score bruto: ${d.valor_bruto_modelo} | Desvio de 0: ${d.desvio_absoluto_zero})${det}`;
    }
</script>
</body>
</html>
"""

# =====================================================================
# 5. ROTAS DA API RESTful E DASHBOARD WEB
# =====================================================================

@app.route('/', methods=['GET'])
def dashboard():
    templates = os.listdir(DIR_TEMPLATES) if os.path.exists(DIR_TEMPLATES) else []
    testes = os.listdir(DIR_TESTES) if os.path.exists(DIR_TESTES) else []
    todas_opcoes_teste = testes + [t for t in templates if t not in testes]
    return render_template_string(HTML_DASHBOARD, templates=templates, testes=todas_opcoes_teste)


@app.route('/predict', methods=['POST'])
def predict():
    if modelo is None and not carregar_modelo():
        return jsonify({"erro": "Modelo não encontrado."}), 500

    dados = request.get_json(silent=True)
    if not dados:
        return jsonify({"erro": "Envie um JSON válido no Body."}), 400

    campos_obrigatorios = ['temperatura', 'vibracao', 'pressao']
    campos_faltantes = [c for c in campos_obrigatorios if c not in dados]
    if campos_faltantes:
        return jsonify({"erro": f"Campos faltando no JSON: {campos_faltantes}"}), 400

    try:
        temp = float(dados['temperatura'])
        vib = float(dados['vibracao'])
        pres = float(dados['pressao'])
    except (ValueError, TypeError):
        return jsonify({"erro": "Os valores de 'temperatura', 'vibracao' e 'pressao' devem ser numéricos."}), 400

    try:
        if features_esperadas is not None:
            linha_dict = {}
            for col in features_esperadas:
                col_lower = col.lower()
                if 'temp' in col_lower:
                    linha_dict[col] = temp
                elif 'vib' in col_lower:
                    linha_dict[col] = vib
                elif 'pres' in col_lower:
                    linha_dict[col] = pres
                else:
                    linha_dict[col] = 0.0
            entrada_modelo = pd.DataFrame([linha_dict], columns=features_esperadas)
        else:
            entrada_modelo = np.array([[temp, vib, pres]])

        if scaler is not None:
            entrada_modelo = scaler.transform(entrada_modelo)

        resultado_bruto = float(modelo.predict(entrada_modelo)[0])
        desvio_zero = abs(resultado_bruto)

        # Lógica de Decisão Corrigida:
        # 1. Mais próximo de 0 = operação normal (dentro da banda de tolerância [-0.5, +0.5]).
        # 2. Se o score fugir para >= 0.5 ou <= -0.5 (ex: -84.2015), ou se houver leitura física impossível/negativa, acusa falha!
        anomalia_fisica = (vib < 0 or pres <= 0 or temp < -20 or temp > 250 or vib > 50 or pres > 300)
        fora_de_tolerancia = (desvio_zero >= 0.5)

        classe_predita = 1 if (fora_de_tolerancia or anomalia_fisica) else 0
        risco_str = "Sim" if classe_predita == 1 else "Não"

        motivo_alerta = "Operação estável (score próximo de 0 e sensores em faixa nominal)."
        if anomalia_fisica and fora_de_tolerancia:
            motivo_alerta = "Alerta Crítico: Score fora da tolerância [-0.5, +0.5] e leitura física extrema/negativa nos sensores!"
        elif fora_de_tolerancia:
            motivo_alerta = f"Alerta: Score do modelo ({resultado_bruto:.4f}) ultrapassou a zona de segurança próxima de zero [-0.5, +0.5]."
        elif anomalia_fisica:
            motivo_alerta = "Alerta Físico: Valores negativos ou fora dos limites operacionais seguros da máquina."

        return jsonify({
            "Risco de Falha": risco_str,
            "mensagem": f"Risco de Falha: {risco_str}",
            "predicao_classe": classe_predita,
            "valor_bruto_modelo": round(resultado_bruto, 4),
            "desvio_absoluto_zero": round(desvio_zero, 4),
            "motivo_alerta": motivo_alerta,
            "leituras_recebidas": {"temperatura": temp, "vibracao": vib, "pressao": pres}
        }), 200
    except Exception as e:
        return jsonify({"erro_interno_modelo": str(e)}), 500


@app.route('/inspect', methods=['POST'])
def inspect_single():
    img_ref = None
    img_teste = None
    nome_saida = "inspecao_customizada.png"

    tol_area = 3.5
    tol_canny = 20.0
    tol_hu = 0.075

    if 'imagem_teste' in request.files:
        arq_teste = request.files['imagem_teste']
        img_teste = ler_imagem_upload(arq_teste)
        nome_saida = f"resultado_{os.path.splitext(arq_teste.filename)[0]}.png"

        tol_area = float(request.form.get('tol_area', 3.5))
        tol_canny = float(request.form.get('tol_canny', 20.0))
        tol_hu = float(request.form.get('tol_hu', 0.075))

        if 'imagem_referencia' in request.files and request.files['imagem_referencia'].filename != '':
            img_ref = ler_imagem_upload(request.files['imagem_referencia'])
        else:
            templates = os.listdir(DIR_TEMPLATES)
            if templates:
                img_ref = ler_imagem_caminho(os.path.join(DIR_TEMPLATES, templates[0]))
    else:
        dados = request.get_json(silent=True) or {}
        nome_teste = dados.get('arquivo_teste')
        templates = os.listdir(DIR_TEMPLATES) if os.path.exists(DIR_TEMPLATES) else []
        nome_ref = dados.get('arquivo_referencia', templates[0] if templates else 'tampa_referencia.jfif')

        tol_area = float(dados.get('tol_area', 3.5))
        tol_canny = float(dados.get('tol_canny', 20.0))
        tol_hu = float(dados.get('tol_hu', 0.075))

        if not nome_teste:
            return jsonify({"erro": "Selecione uma imagem de teste."}), 400

        img_ref = ler_imagem_caminho(os.path.join(DIR_TEMPLATES, nome_ref))
        caminho_t = os.path.join(DIR_TESTES, nome_teste)
        if not os.path.exists(caminho_t):
            caminho_t = os.path.join(DIR_TEMPLATES, nome_teste)
        img_teste = ler_imagem_caminho(caminho_t)
        nome_saida = f"resultado_{os.path.splitext(nome_teste)[0]}.png"

    if img_ref is None or img_teste is None:
        return jsonify({"erro": "Imagem de referência ou de teste inválida."}), 404

    resultado = comparar_pecas(img_ref, img_teste, nome_saida=nome_saida,
                               tol_area=tol_area, tol_canny=tol_canny, tol_hu=tol_hu)
    return jsonify(resultado), 200


@app.route('/inspect/batch', methods=['GET'])
def inspect_batch():
    templates = os.listdir(DIR_TEMPLATES) if os.path.exists(DIR_TEMPLATES) else []
    if not templates:
        return jsonify({"erro": "Nenhuma imagem em assets/templates."}), 404

    img_ref = ler_imagem_caminho(os.path.join(DIR_TEMPLATES, templates[0]))
    arquivos_teste = os.listdir(DIR_TESTES) if os.path.exists(DIR_TESTES) else []
    relatorio = []

    for nome_arq in arquivos_teste:
        img_t = ler_imagem_caminho(os.path.join(DIR_TESTES, nome_arq))
        if img_t is not None:
            res = comparar_pecas(img_ref, img_t, nome_saida=f"diag_{os.path.splitext(nome_arq)[0]}.png")
            res.pop("imagem_base64", None)
            res["arquivo_analisado"] = nome_arq
            relatorio.append(res)

    return jsonify({
        "peca_referencia_utilizada": templates[0],
        "total_pecas_analisadas": len(relatorio),
        "resultados": relatorio
    }), 200


if __name__ == '__main__':
    print("=========================================================")
    print(" Painel Web MetalTech:  http://127.0.0.1:5000")
    print(" Endpoint Preditivo:    http://127.0.0.1:5000/predict")
    print(" Endpoint Visão Lote:   http://127.0.0.1:5000/inspect/batch")
    print("=========================================================")
    app.run(host='0.0.0.0', port=5000, debug=True)