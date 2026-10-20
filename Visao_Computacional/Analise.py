import os
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
CAMINHO_CSV_LIMPO = os.path.join(DIR_RAIZ, 'Machine_Learning', 'Dados_CSV', 'dados_limpos.csv')

# =====================================================================
# 2. CARREGAMENTO INTELIGENTE DO MODELO E SCALER
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

        # Descobre automaticamente quais colunas o modelo aprendeu no Tratamento.ipynb
        if hasattr(modelo, 'feature_names_in_'):
            features_esperadas = list(modelo.feature_names_in_)
        if hasattr(modelo, 'n_features_in_'):
            n_features = int(modelo.n_features_in_)

        print(f"[SUCESSO] Modelo carregado: {type(modelo).__name__}")
        print(f"[INFO] Quantidade de features esperadas pelo modelo: {n_features}")
        if features_esperadas:
            print(f"[INFO] Nomes das colunas esperadas: {features_esperadas}")
        return True
    except Exception as e:
        print(f"[ERRO] Falha ao carregar ficheiro.joblib: {e}")
        return False

carregar_modelo()

# =====================================================================
# 3. ROTAS DA API RESTful
# =====================================================================

@app.route('/', methods=['GET'])
def health_check():
    """Rota para inspecionar o que o modelo espera exatamente."""
    return jsonify({
        "servico": "API MetalTech Soluções Industriais S.A.",
        "status_api": "Online",
        "tipo_modelo": type(modelo).__name__ if modelo is not None else None,
        "n_features_esperadas": n_features,
        "colunas_esperadas_pelo_modelo": features_esperadas
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    if modelo is None:
        if not carregar_modelo():
            return jsonify({"erro": "Modelo não encontrado."}), 500

    dados = request.get_json(silent=True)
    if not dados:
        return jsonify({"erro": "Envie um JSON válido no Body."}), 400

    campos_obrigatorios = ['temperatura', 'vibracao', 'pressao']
    campos_faltantes = [c for c in campos_obrigatorios if c not in dados]
    if campos_faltantes:
        return jsonify({"erro": f"Campos faltando no JSON: {campos_faltantes}"}), 400

    # 1. Validação EXCLUSIVA para saber se os dados são numéricos
    try:
        temp = float(dados['temperatura'])
        vib = float(dados['vibracao'])
        pres = float(dados['pressao'])
    except (ValueError, TypeError):
        return jsonify({
            "erro": "Os valores de 'temperatura', 'vibracao' e 'pressao' devem ser numéricos."
        }), 400

    # 2. Montagem inteligente do vetor de entrada e Predição
    try:
        valores_base = [temp, vib, pres]

        # Se o modelo foi treinado com mais de 3 colunas (ex: id ou coluna extra do CSV),
        # preenchemos o restante com 0.0 ou mapeamos pelo nome para não quebrar
        if features_esperadas is not None:
            linha_dict = {}
            for i, col in enumerate(features_esperadas):
                col_lower = col.lower()
                if 'temp' in col_lower:
                    linha_dict[col] = temp
                elif 'vib' in col_lower:
                    linha_dict[col] = vib
                elif 'pres' in col_lower:
                    linha_dict[col] = pres
                elif col in dados:
                    linha_dict[col] = float(dados[col])
                else:
                    # Caso exista alguma 4ª coluna no treino, preenche com 0 ou valor neutro
                    linha_dict[col] = 0.0
            entrada_modelo = pd.DataFrame([linha_dict], columns=features_esperadas)
        else:
            # Se o modelo foi treinado com array NumPy puro (ex: apos StandardScaler)
            if len(valores_base) < n_features:
                valores_base.extend([0.0] * (n_features - len(valores_base)))
            entrada_modelo = np.array([valores_base[:n_features]])

        # Aplica o scaler se existir
        if scaler is not None:
            entrada_modelo = scaler.transform(entrada_modelo)

        # Faz a predição (funciona tanto para Classificador quanto para LinearRegression)
        resultado_bruto = float(modelo.predict(entrada_modelo)[0])

        # Se for Regressão Linear, valores >= 0.5 indicam falha (1); se for Classificador, já é 0 ou 1
        classe_predita = 1 if resultado_bruto >= 0.5 else 0
        risco_str = "Sim" if classe_predita == 1 else "Não"

        resposta = {
            "Risco de Falha": risco_str,
            "mensagem": f"Risco de Falha: {risco_str}",
            "predicao_classe": classe_predita,
            "valor_bruto_modelo": round(resultado_bruto, 4),
            "leituras_recebidas": {
                "temperatura": temp,
                "vibracao": vib,
                "pressao": pres
            }
        }

        if hasattr(modelo, "predict_proba"):
            prob = float(modelo.predict_proba(entrada_modelo)[0][1])
            resposta["probabilidade_falha"] = round(prob, 4)

        return jsonify(resposta), 200

    except Exception as e:
        # Agora, se o Scikit-Learn reclamar de algo, ele mostrará a mensagem real!
        return jsonify({
            "erro_interno_modelo": str(e),
            "tipo_modelo": type(modelo).__name__,
            "n_features_esperadas": n_features,
            "colunas_esperadas": features_esperadas
        }), 500


if __name__ == '__main__':
    print("====================================================")
    print(" Iniciando Servidor Flask - MetalTech Etapa 3")
    print(" Endpoint Preditivo: http://127.0.0.1:5000/predict")
    print("====================================================")
    app.run(host='0.0.0.0', port=5000, debug=True)