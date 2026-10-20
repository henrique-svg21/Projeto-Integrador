# Projeto Integrador: Fábrica Inteligente — MetalTech Soluções Industriais S.A.

Protótipo industrial desenvolvido para integrar **Machine Learning Preditivo (IoT)** e **Visão Computacional (OpenCV)** em uma arquitetura de microsserviços RESTful com Flask.

---

## 📌 Visão Geral do Sistema

O projeto resolve dois desafios operacionais da fábrica:

1. **Inteligência Preditiva:** analisa dados de sensores de telemetria (temperatura, vibração e pressão) para antecipar falhas críticas em motores industriais.
2. **Inspeção Visual Automatizada:** compara imagens de peças produzidas (ex.: tampas de motores) com uma peça-gabarito de referência, detectando quebras, trincas superficiais, obstruções no furo central e variações geométricas, de forma invariante à rotação.

---

## 🛠️ Tecnologias e Bibliotecas Utilizadas

O sistema requer **Python 3.9 ou superior** e os seguintes pacotes:

- **Manipulação e Engenharia de Dados:** `pandas`, `numpy`
- **Machine Learning e Serialização:** `scikit-learn`, `joblib`
- **Visão Computacional e Processamento Digital de Imagens:** `opencv-python`
- **Backend Web e API RESTful:** `flask`
- **Ambiente de Desenvolvimento Interativo:** `jupyter`, `ipykernel`

### Instalação das Dependências

Para instalar todas as dependências, execute o seguinte comando no terminal:

```bash
pip install pandas numpy scikit-learn joblib opencv-python flask jupyter ipykernel
```

---

## 📂 Estrutura do Repositório

```text
PROJETO-INTEGRADOR/
│
├── Machine_Learning/
│   ├── Code/
│   │   └── Tratamento.ipynb             # Notebook de EDA, Data Cleaning e Treinamento
│   ├── Dados_CSV/
│   │   ├── dados_sensores_metaltech.csv  # Dados brutos de IoT
│   │   └── dados_limpos.csv              # Gerado após a execução do notebook
│   └── Registro/
│       ├── ficheiro.joblib               # Modelo exportado (gerado pelo notebook)
│       └── MLOps.md                      # Planejamento de monitoramento contínuo
│
├── Visao_Computacional/
│   ├── assets/
│   │   ├── templates/                    # Peças de referência
│   │   └── testes/                       # Peças de teste (boas, trincadas e quebradas)
│   ├── output/                           # Imagens de diagnóstico geradas com filtros
│   └── Analise.py                        # Servidor Flask com painel web e endpoints da API
│
├── .gitignore
├── LICENSE
└── README.md
```

> **Atenção:** os arquivos `Machine_Learning/Dados_CSV/dados_limpos.csv` e `Machine_Learning/Registro/ficheiro.joblib` estão no `.gitignore` e, portanto, não constam no repositório remoto. Eles serão criados localmente ao seguir a ordem de execução descrita abaixo.

---

## 🚀 Ordem de Execução e Testes

Para garantir que a API encontre o modelo preditivo, execute as etapas exatamente nesta ordem.

### Passo 1: Executar o Treinamento do Modelo (Machine Learning)

1. Abra o arquivo `Machine_Learning/Code/Tratamento.ipynb` no VS Code ou no Jupyter Notebook.
2. Execute todas as células do notebook.
3. Ao finalizar, certifique-se de que os seguintes arquivos foram criados localmente:
   - `Machine_Learning/Dados_CSV/dados_limpos.csv`
   - `Machine_Learning/Registro/ficheiro.joblib`

### Passo 2: Iniciar a API e o Painel Web (Flask)

Abra o terminal na pasta raiz do repositório (`PROJETO-INTEGRADOR`) e inicie o servidor com o seguinte comando:

```bash
python Visao_Computacional/Analise.py
```

O servidor será iniciado localmente, disponibilizando as seguintes rotas:

- **Painel Web Gráfico:** http://127.0.0.1:5000
- **Endpoint Preditivo (POST):** http://127.0.0.1:5000/predict
- **Inspeção em Lote (GET):** http://127.0.0.1:5000/inspect/batch

### Passo 3: Como Testar o Sistema

#### 1. Teste Visual pelo Navegador (Recomendado)

Acesse http://127.0.0.1:5000 em qualquer navegador.

- **Seleção de Peças:** escolha a peça de referência e a peça a ser analisada no menu suspenso ou utilize a aba de upload para enviar uma imagem do computador.
- **Sliders de Tolerância:** ajuste dinamicamente a tolerância de área (quebras), Canny (trincas) e momentos de Hu (geometria).
- **Diagnóstico Instantâneo:** clique em **"Analisar Peça Agora"** para visualizar o laudo técnico (aprovada/reprovada) e a comparação lado a lado da peça com o filtro Canny Edge Detection.
- **Simulador IoT:** preencha os valores de temperatura, vibração e pressão e clique em **"Prever Risco do Motor"** para testar o modelo preditivo em tempo real.

#### 2. Teste via Postman ou Insomnia (Rota `/predict`)

Configure a requisição com os seguintes parâmetros:

- **Método:** `POST`
- **URL:** `http://127.0.0.1:5000/predict`
- **Header:** `Content-Type: application/json`
- **Corpo:** selecione `raw` e o formato `JSON`, utilizando o seguinte conteúdo:

```json
{
  "temperatura": 88.5,
  "vibracao": 5.2,
  "pressao": 110.0
}
```

**Retorno esperado (`200 OK`):**

```json
{
  "Risco de Falha": "Não",
  "mensagem": "Risco de Falha: Não",
  "predicao_classe": 0,
  "valor_bruto_modelo": 0.0821,
  "leituras_recebidas": {
    "temperatura": 88.5,
    "vibracao": 5.2,
    "pressao": 110.0
  }
}
```

> **Observação:** o retorno acima é um exemplo ilustrativo. Os valores retornados podem variar conforme o modelo treinado e os dados enviados.

#### 3. Teste em Lote via Postman (Rota `/inspect/batch`)

Configure a requisição com os seguintes parâmetros:

- **Método:** `GET`
- **URL:** `http://127.0.0.1:5000/inspect/batch`

Essa rota executa a inspeção comparativa automática de todos os arquivos contidos em `Visao_Computacional/assets/testes/` contra o gabarito `Visao_Computacional/assets/templates/tampa_referencia.jfif`.

As imagens anotadas geradas durante o processo são salvas na pasta `Visao_Computacional/output/`.

---

## 📋 Resumo do Fluxo de Execução

1. Instalar as dependências do projeto.
2. Executar todas as células do notebook `Tratamento.ipynb` para gerar os dados tratados e o modelo preditivo.
3. Iniciar o servidor Flask por meio do arquivo `Analise.py`.
4. Acessar o painel web ou testar os endpoints da API.
5. Consultar as imagens de diagnóstico geradas na pasta `Visao_Computacional/output/`.
