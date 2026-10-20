 Um plano de MLOps (Machine Learning Operations) estrutura as práticas para levar modelos de Inteligência Artificial do ambiente de experimentos até a produção de forma automatizada, segura e escalável. Aqui, treinamos os modelos de regressão linear e regressão floresta, testando sua eficácia em identificar as métricas de Acurácia, Precisão, F1-Score e Recall.

Nos testes realizados, foi possível identificar que o modelo de regressão linear teve uma ótima acurácia (95.1%), enquanto o de regressão floresta se saiu ainda melhor, com 99.95% de acurácia. Entretanto, quando observamos a precisão de ambos, temos uma diferença extrema: Enquanto o modelo de floresta alcançou 98.99% de precisão, o de regressão linear alcançou 0% de precisão. 

Isso acontece porque o dataset da fábrica apresenta um forte desbalanceamento de classes (cerca de 95% dos dados são de operação normal e apenas 5% são falhas). Por ser um modelo linear contínuo, a Regressão Linear "chutou" que todos os registros eram de operação normal. Como 95% do banco realmente era normal, ela obteve 95,1% de acurácia, mas obteve 0% de Precisão e 0% de Recall, pois não detectou nenhuma falha real.

Por outro lado, o algoritmo Random Forest (Floresta Aleatória) capturou com precisão as relações não-lineares dos sensores, atingindo 100% de Recall e 98,99% de Precisão. Por garantir que nenhuma quebra de motor passe despercebida na linha de produção, o Random Forest foi o modelo selecionado para ir para a produção.

Para garantir que o modelo selecionado (Random Forest) não perca capacidade preditiva ao longo do tempo devido ao desgaste natural das máquinas ou mudanças no ambiente de fábrica, estabelece-se o seguinte plano de MLOps para os próximos 6 meses:

#### 1. Versionamento e Registro do Modelo (Mês 1)
* **Artefatos:** Versionar e armazenar o arquivo do modelo (`.joblib`) e as rotinas de pré-processamento/limpeza de dados em repositório centralizado.
* **Logs de Produção:** Configurar a API REST para registrar cada requisição dos sensores IoT (temperatura, vibração, pressão) e a previsão gerada em um banco de dados de logs.

#### 2. Monitoramento de Degradação (Data Drift e Concept Drift) (Meses 1 a 6)
* **Monitoramento de Data Drift (Mudança nos Dados):** Acompanhar semanalmente a distribuição estatística das variáveis numéricas. Se a vibração média dos motores aumentar devido ao desgaste mecânico normal das peças, o modelo deve ser avisado para não emitir alarmes falsos em excesso.
* **Métrica Crítica de Alerta (Recall):** O **Recall** será a métrica principal de monitoramento. Se a taxa de Recall cair abaixo de **95%** na verificação com as paradas reais da fábrica, o sistema emitirá um alerta automático para a equipe de engenharia de dados.

#### 3. Re-treinamento e Governança (Trimestral / Semestral)
* **Ciclo de Re-treinamento (Mês 3 e Mês 6):** Alimentar o pipeline de treinamento com os novos dados de sensores acumulados nos últimos 90 dias para atualizar os padrões das árvores de decisão do Random Forest, visto que uma análise feita com dados desatualizados tende a ser pouco precisa.
* **Mecanismo de Rollback:** Antes de implantar uma nova versão do modelo em produção, um teste automatizado comparará o novo modelo com a versão atual. Se a nova versão apresentar desempenho inferior no Recall, o sistema executa um *rollback* automático para a versão anterior.