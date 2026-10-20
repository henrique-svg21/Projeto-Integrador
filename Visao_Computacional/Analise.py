import os
import sys
import cv2
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

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
n_features = 3

def carregar_modelo():
    global modelo, scaler, features_esperadas, n_features
    if not os.path.exists(CAMINHO_MODELO):
        print(f"[AVISO] Arquivo não encontrado em: {CAMINHO_MODELO}")
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
        if hasattr(modelo, 'n_features_in_'):
            n_features = int(modelo.n_features_in_)

        print(f"[SUCESSO] Modelo carregado: {type(modelo).__name__}")
        if features_esperadas:
            print(f"[INFO] Nomes das colunas esperadas: {features_esperadas}")
        return True
    except Exception as e:
        print(f"[ERRO] Falha ao carregar ficheiro.joblib: {e}")
        return False

carregar_modelo()

# =====================================================================
# 3. MOTOR DE VISÃO COMPUTACIONAL (OPENCV - ETAPA 3)
# =====================================================================

def ler_imagem_caminho(caminho):
    """Lê imagens (.jfif, .png, .jpg) suportando caminhos com caracteres especiais no Windows."""
    if not os.path.exists(caminho):
        return None
    bytes_img = np.fromfile(caminho, dtype=np.uint8)
    return cv2.imdecode(bytes_img, cv2.IMREAD_COLOR)

def ler_imagem_upload(arquivo_storage):
    """Lê imagem enviada diretamente via upload (form-data) no Postman/Insomnia."""
    bytes_img = np.frombuffer(arquivo_storage.read(), np.uint8)
    return cv2.imdecode(bytes_img, cv2.IMREAD_COLOR)

def salvar_imagem_seguro(caminho, img):
    """Salva a imagem processada na pasta output."""
    sucesso, buffer = cv2.imencode('.png', img)
    if sucesso:
        buffer.tofile(caminho)

def extrair_caracteristicas_peca(img_bgr, dimensao=(500, 500)):
    """
    Aplica o pipeline exigido na Etapa 3:
    1. Redimensionamento padronizado
    2. Escala de Cinza (Grayscale)
    3. Filtro Blur (GaussianBlur)
    4. Canny Edge Detection
    5. Extração de Contornos e Métricas Geométricas Invariantes à Rotação
    """
    img_res = cv2.resize(img_bgr, dimensao)
    
    # 1. Escala de Cinza e 2. Gaussian Blur
    cinza = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(cinza, (5, 5), 0)

    # Segmentação do fundo (detecta automaticamente se o fundo é claro ou escuro)
    cantos = [blur[0, 0], blur[0, -1], blur[-1, 0], blur[-1, -1]]
    fundo_claro = np.mean(cantos) > 127
    tipo_thresh = cv2.THRESH_BINARY_INV if fundo_claro else cv2.THRESH_BINARY
    _, mascara = cv2.threshold(blur, 0, 255, tipo_thresh + cv2.THRESH_OTSU)

    # Limpeza morfológica de pequenos ruídos
    kernel = np.ones((3, 3), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=1)

    # 3. Canny Edge Detection (exigido no PDF)
    bordas_canny = cv2.Canny(blur, 50, 150)

    # 4. Extração de Contornos com Hierarquia (identifica contorno externo e furos internos)
    contornos, hierarquia = cv2.findContours(mascara, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if not contornos:
        return None

    # Maior contorno = silhueta principal da peça
    idx_maior = int(np.argmax([cv2.contourArea(c) for c in contornos]))
    contorno_principal = contornos[idx_maior]

    # Centraliza a peça usando o bounding box para tornar a comparação invariante à posição
    x, y, w, h = cv2.boundingRect(contorno_principal)
    recorte_mascara = cv2.resize(mascara[y:y+h, x:x+w], (400, 400))
    recorte_canny = cv2.resize(bordas_canny[y:y+h, x:x+w], (400, 400))

    # Métricas extraídas
    area_solida = float(np.sum(recorte_mascara > 0))
    densidade_canny = float(np.sum(recorte_canny > 0))
    perimetro = float(cv2.arcLength(contorno_principal, True))

    # Contagem e área de furos internos (filhos do contorno principal na hierarquia)
    furos_internos = 0
    area_furos = 0.0
    if hierarquia is not None:
        for i, c in enumerate(contornos):
            if hierarquia[0][i][3] == idx_maior and cv2.contourArea(c) > 80:
                furos_internos += 1
                area_furos += cv2.contourArea(c)

    return {
        "img_redimensionada": img_res,
        "mascara": mascara,
        "canny": bordas_canny,
        "contornos": contornos,
        "contorno_principal": contorno_principal,
        "area_solida": area_solida,
        "densidade_canny": densidade_canny,
        "perimetro": perimetro,
        "furos_internos": furos_internos,
        "area_furos": area_furos
    }


def comparar_pecas(img_ref, img_teste, nome_saida="resultado_inspecao.png"):
    """Compara a peça de referência com a peça de teste e gera laudo + imagem anotada."""
    feat_ref = extrair_caracteristicas_peca(img_ref)
    feat_teste = extrair_caracteristicas_peca(img_teste)

    if feat_ref is None or feat_teste is None:
        return {"erro": "Não foi possível detectar contornos válidos em uma das imagens."}

    # 1. Diferença percentual de área sólida (pega falta de orelha ou obstrução no furo)
    diff_area_pct = abs(feat_teste["area_solida"] - feat_ref["area_solida"]) / (feat_ref["area_solida"] + 1e-5) * 100

    # 2. Diferença de densidade de bordas Canny (pega fissuras/trincas e desgastes superficiais)
    diff_canny_pct = abs(feat_teste["densidade_canny"] - feat_ref["densidade_canny"]) / (feat_ref["densidade_canny"] + 1e-5) * 100

    # 3. Distância de forma invariante à rotação (Hu Moments via matchShapes)
    dist_forma_hu = float(cv2.matchShapes(feat_ref["contorno_principal"], feat_teste["contorno_principal"], cv2.CONTOURS_MATCH_I1, 0.0))

    # 4. Diferença na área dos furos internos (pega furo central obstruído ou orelha quebrada)
    diff_furos_pct = abs(feat_teste["area_furos"] - feat_ref["area_furos"]) / (feat_ref["area_furos"] + 1.0) * 100

    # Regras de Decisão Industrial
    motivos_reprovacao = []

    if diff_area_pct > 2.2:
        motivos_reprovacao.append(f"Anomalia na área estrutural da peça (variação de {diff_area_pct:.2f}% - possível quebra ou obstrução)")
    
    if diff_furos_pct > 15.0 or feat_teste["furos_internos"] != feat_ref["furos_internos"]:
        motivos_reprovacao.append(
            f"Alteração nas cavidades/furos internos (Esperado: {feat_ref['furos_internos']} furos | Detectado: {feat_teste['furos_internos']}, variação de área: {diff_furos_pct:.1f}%)"
        )

    if diff_canny_pct > 6.5:
        motivos_reprovacao.append(f"Descontinuidade superficial nas bordas Canny (variação de {diff_canny_pct:.2f}% - possível fissura, trinca ou rebarba)")

    if dist_forma_hu > 0.08:
        motivos_reprovacao.append(f"Deformação geométrica do contorno principal (Índice Hu: {dist_forma_hu:.4f})")

    aprovada = len(motivos_reprovacao) == 0
    status_str = "APROVADA" if aprovada else "REPROVADA"
    cor_status = (0, 200, 0) if aprovada else (0, 0, 255)

    # Geração do painel visual (Imagem Original + Contornos | Filtro Canny)
    vis_contornos = feat_teste["img_redimensionada"].copy()
    cv2.drawContours(vis_contornos, feat_teste["contornos"], -1, cor_status, 2)
    cv2.putText(vis_contornos, f"Status: {status_str}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor_status, 2)
    cv2.putText(vis_contornos, f"Diff Area: {diff_area_pct:.1f}% | Diff Canny: {diff_canny_pct:.1f}%", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

    canny_bgr = cv2.cvtColor(feat_teste["canny"], cv2.COLOR_GRAY2BGR)
    cv2.putText(canny_bgr, "Filtro Canny Edge", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    painel_final = np.hstack((vis_contornos, canny_bgr))
    caminho_salvo = os.path.join(DIR_OUTPUT, nome_saida)
    salvar_imagem_seguro(caminho_salvo, painel_final)

    return {
        "status_qualidade": status_str,
        "peca_aprovada": aprovada,
        "motivos_reprovacao": motivos_reprovacao if not aprovada else ["Geometria e superfície em conformidade com o padrão (invariante à rotação)."],
        "metricas_comparativas": {
            "variacao_area_percentual": round(diff_area_pct, 2),
            "variacao_bordas_canny_percentual": round(diff_canny_pct, 2),
            "distancia_geometrica_hu": round(dist_forma_hu, 4),
            "furos_esperados": feat_ref["furos_internos"],
            "furos_detectados": feat_teste["furos_internos"],
            "variacao_area_furos_percentual": round(diff_furos_pct, 2)
        },
        "imagem_diagnostico_salva_em": caminho_salvo
    }

# =====================================================================
# 4. ROTAS DA API RESTful (PREDITIVA + VISÃO COMPUTACIONAL)
# =====================================================================

@app.route('/', methods=['GET'])
def health_check():
    templates_disp = os.listdir(DIR_TEMPLATES) if os.path.exists(DIR_TEMPLATES) else []
    testes_disp = os.listdir(DIR_TESTES) if os.path.exists(DIR_TESTES) else []
    return jsonify({
        "servico": "API MetalTech Soluções Industriais S.A.",
        "status_api": "Online",
        "rotas_disponiveis": {
            "POST /predict": "Predição de falha de motor via JSON (temperatura, vibracao, pressao)",
            "POST /inspect": "Inspeção visual comparando peça de teste com referência (aceita Upload form-data ou JSON)",
            "GET /inspect/batch": "Executa a inspeção em todas as imagens da pasta assets/testes automaticamente"
        },
        "arquivos_templates": templates_disp,
        "arquivos_testes": testes_disp
    }), 200


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
        classe_predita = 1 if resultado_bruto >= 0.5 else 0
        risco_str = "Sim" if classe_predita == 1 else "Não"

        return jsonify({
            "Risco de Falha": risco_str,
            "mensagem": f"Risco de Falha: {risco_str}",
            "predicao_classe": classe_predita,
            "valor_bruto_modelo": round(resultado_bruto, 4),
            "leituras_recebidas": {"temperatura": temp, "vibracao": vib, "pressao": pres}
        }), 200
    except Exception as e:
        return jsonify({"erro_interno_modelo": str(e)}), 500


@app.route('/inspect', methods=['POST'])
def inspect_single():
    """
    Permite ao usuário comparar peças de 2 formas:
    OPÇÃO A (Upload de arquivos via form-data no Postman):
      - Envie 'imagem_teste' (obrigatório) e 'imagem_referencia' (opcional; se omitido, usa tampa_referencia.jfif)
    OPÇÃO B (Via JSON escolhendo arquivos da pasta assets):
      - {"arquivo_teste": "tampa_fissurada.jfif", "arquivo_referencia": "tampa_referencia.jfif"}
    """
    img_ref = None
    img_teste = None
    nome_saida = "inspecao_customizada.png"

    # Caso 1: Usuário enviou arquivos via Upload (multipart/form-data)
    if 'imagem_teste' in request.files:
        arq_teste = request.files['imagem_teste']
        img_teste = ler_imagem_upload(arq_teste)
        nome_saida = f"resultado_{os.path.splitext(arq_teste.filename)[0]}.png"

        if 'imagem_referencia' in request.files:
            img_ref = ler_imagem_upload(request.files['imagem_referencia'])
        else:
            # Usa a primeira imagem da pasta templates como padrão
            templates = os.listdir(DIR_TEMPLATES)
            if templates:
                img_ref = ler_imagem_caminho(os.path.join(DIR_TEMPLATES, templates[0]))

    # Caso 2: Usuário enviou os nomes dos arquivos via JSON
    else:
        dados = request.get_json(silent=True) or {}
        nome_teste = dados.get('arquivo_teste')
        templates = os.listdir(DIR_TEMPLATES) if os.path.exists(DIR_TEMPLATES) else []
        nome_ref = dados.get('arquivo_referencia', templates[0] if templates else 'tampa_referencia.jfif')

        if not nome_teste:
            return jsonify({
                "erro": "Envie 'imagem_teste' via form-data (File) OU 'arquivo_teste' via JSON.",
                "exemplos_disponiveis": os.listdir(DIR_TESTES) if os.path.exists(DIR_TESTES) else []
            }), 400

        img_ref = ler_imagem_caminho(os.path.join(DIR_TEMPLATES, nome_ref))
        img_teste = ler_imagem_caminho(os.path.join(DIR_TESTES, nome_teste))
        nome_saida = f"resultado_{os.path.splitext(nome_teste)[0]}.png"

    if img_ref is None:
        return jsonify({"erro": "Imagem de referência não encontrada ou inválida."}), 404
    if img_teste is None:
        return jsonify({"erro": "Imagem de teste não encontrada ou inválida."}), 404

    resultado = comparar_pecas(img_ref, img_teste, nome_saida=nome_saida)
    return jsonify(resultado), 200


@app.route('/inspect/batch', methods=['GET'])
def inspect_batch():
    """Roda a inspeção em todas as imagens da pasta assets/testes comparando com tampa_referencia.jfif."""
    templates = os.listdir(DIR_TEMPLATES) if os.path.exists(DIR_TEMPLATES) else []
    if not templates:
        return jsonify({"erro": "Nenhuma imagem encontrada em assets/templates."}), 404

    caminho_ref = os.path.join(DIR_TEMPLATES, templates[0])
    img_ref = ler_imagem_caminho(caminho_ref)

    arquivos_teste = os.listdir(DIR_TESTES) if os.path.exists(DIR_TESTES) else []
    relatorio = []

    for nome_arq in arquivos_teste:
        caminho_t = os.path.join(DIR_TESTES, nome_arq)
        img_t = ler_imagem_caminho(caminho_t)
        if img_t is not None:
            nome_out = f"diag_{os.path.splitext(nome_arq)[0]}.png"
            res = comparar_pecas(img_ref, img_t, nome_saida=nome_out)
            res["arquivo_analisado"] = nome_arq
            relatorio.append(res)

    return jsonify({
        "peca_referencia_utilizada": templates[0],
        "total_pecas_analisadas": len(relatorio),
        "resultados": relatorio
    }), 200


if __name__ == '__main__':
    print("====================================================")
    print(" Iniciando Servidor Flask - MetalTech Etapa 3")
    print(" 1. Preditivo (POST):    http://127.0.0.1:5000/predict")
    print(" 2. Visão Única (POST):  http://127.0.0.1:5000/inspect")
    print(" 3. Visão Lote (GET):    http://127.0.0.1:5000/inspect/batch")
    print("====================================================")
    app.run(host='0.0.0.0', port=5000, debug=True)