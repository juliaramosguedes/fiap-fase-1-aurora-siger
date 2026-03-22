# 🚀 Missão Aurora Siger

## Relatório Operacional de Pré-Decolagem

*Atividade Integradora — Fase 1 | Ciência da Computação — FIAP*

**Julia Ramos · Carlos Eugenio · Julio Guma · Matheus Fuchens**

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?logo=numpy&logoColor=white) ![pandas](https://img.shields.io/badge/Pandas-2.0+-150458?logo=pandas&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-1.0-F37626?logo=jupyter&logoColor=white) ![scikit-learn](https://img.shields.io/badge/Scikit--Learn-1.3-F7931E?logo=scikit-learn&logoColor=white)

---

## Sobre o projeto

Aurora Siger simula o sistema de verificação de pré-decolagem de uma nave espacial interplanetária. Antes de qualquer lançamento real, engenheiros executam uma checklist técnica para confirmar que todos os subsistemas estão dentro dos limites operacionais seguros. Este projeto replica esse processo em software.

O sistema recebe dados de telemetria da nave e produz uma decisão binária:

```
PRONTO PARA DECOLAR   ou   DECOLAGEM ABORTADA
```

Além da verificação determinística, dois modelos de machine learning analisam se a leitura atual se parece com missões históricas nominais ou com missões que apresentaram anomalias — e emitem uma avaliação de risco.

---

## Como funciona

A pipeline segue uma sequência onde cada etapa depende da anterior:

```
Telemetria → Limiares de segurança → Verificações → Análise energética → Decisão → IA
```

**Telemetria** — os dados de entrada: temperaturas interna e externa, integridade estrutural (flag binário 0/1), nível de energia (%), pressão dos tanques e status dos 5 subsistemas críticos.

**Limiares de segurança** — constantes nomeadas derivadas de dados reais (MIT OCW, ESA Bulletin 87, ESA Advanced Concepts Team). São a **única fonte de verdade** do sistema: usadas tanto nas verificações quanto no treinamento dos modelos de IA.

**Verificações** — 7 funções puras, uma por parâmetro, cada uma retornando `bool`. A função `verify_launch_readiness` consolida tudo e emite a decisão final.

**Análise energética** — calcula carga atual, perdas resistivas (P = I²R), eficiência do sistema (η) e autonomia em horas com base nos parâmetros da nave.

**Análise por IA** — dois modelos com abordagens complementares, escolhidos deliberadamente:

- **IsolationForest** (não supervisionado): aprende o padrão nominal e detecta qualquer desvio sem precisar de anomalias históricas rotuladas. Em operações espaciais, anomalias rotuladas são raras — esse modelo é mais robusto nesse cenário. Atingiu **recall de 94,8%**.
- **DecisionTreeClassifier** (supervisionado): aprende regras if-then a partir dos dados históricos — a mesma estrutura lógica do checklist da Seção 3. O modelo redescobre autonomamente os limiares de segurança codificados à mão, criando uma ponte direta entre a abordagem determinística e a abordagem de Machine Learning. Atingiu **recall de 99,8% (CV 5-fold)**.

Usar os dois e comparar o consenso é mais confiável do que depender de um único modelo. Em sistemas de segurança crítica, **recall é a métrica prioritária**: um falso negativo — anomalia não detectada — é mais perigoso que um abort desnecessário.

---

## Decisões técnicas

**Programação funcional** — todas as funções de verificação são puras: sem estado interno, sem efeitos colaterais, mesmo input sempre produz mesmo output. Isso torna cada função testável individualmente — mandatório em sistemas de segurança crítica.

**Imutabilidade** — `TelemetryReading` e `EnergyAnalysis` são dataclasses frozen. Representam uma leitura num momento específico do tempo (T-0). Não faz sentido modificar dados de telemetria após a leitura.

**Single source of truth** — os limiares de segurança são definidos uma vez na Seção 3 e reutilizados no dataset de treinamento da IA. Se o limite de energia mínima muda de 60% para 65%, muda em um lugar e propaga para o sistema inteiro.

**Dados de referência reais** — os limiares operacionais vêm de documentação técnica da NASA e ESA. Valores sem dataset de origem são marcados como `# SIMULATED` no código e documentados em `telemetry_reference.md`.

**Escolha dos modelos de IA** — o padrão de anomalia neste problema é uma disjunção de condições por feature (OR de thresholds). IsolationForest detecta isso estatisticamente, sem regras explícitas. DecisionTreeClassifier aprende exatamente esse tipo de fronteira — geometricamente compatível com o problema. Se os limiares foram bem definidos, a árvore os redescobre: é uma validação indireta do checklist.

---

## Resultado da execução

```
============================================================
AURORA SIGER — PRE-LAUNCH TELEMETRY SNAPSHOT
============================================================
  Internal temperature :     22.5 °C
  External temperature :    -45.0 °C
  Structural integrity :        1   (1=OK, 0=FAIL)
  Energy level         :     87.3 %
  Tank pressure        :     34.8 bar
  Propulsion system    :       OK
  Power management     :       OK
  Communications       :       OK
  Thermal control      :       OK
  Navigation / ADCS    :       OK
============================================================
============================================================
PRE-LAUNCH VERIFICATION CHECKLIST
============================================================
  ✅  Internal Temperature           PASS
  ✅  External Temperature           PASS
  ✅  Structural Integrity           PASS
  ✅  Energy Level                   PASS
  ✅  Tank Pressure                  PASS
  ✅  Critical Modules               PASS
  ✅  Autonomy                       PASS
============================================================

  >>> PRONTO PARA DECOLAR <<<

============================================================
AI RISK ASSESSMENT — AURORA SIGER
============================================================
  Data classification
    IsolationForest (unsupervised) : NOMINAL
    DecisionTree    (supervised)   : NOMINAL
    Model consensus               : YES

  Anomaly identification
    Anomaly probability           : 0.0%
    IsolationForest recall        : 94.8%
    DecisionTree recall           : 100.0%  (CV 5-fold: 99.8% ± 0.4%)

  Risk suggestion
    Risk level                    : LOW
    Recommended action            : PROCEED WITH CAUTION
============================================================
```

> **Por que a DecisionTree atinge 100% de recall?** A árvore aprende regras if-then — a mesma estrutura lógica do checklist. Em dados sintéticos gerados a partir de limites fixos, ela redescobre os próprios limiares codificados na Seção 3 (ex.: `structural_integrity <= 0.5 → ANOMALIA`, `energy_pct <= 62 → ANOMALIA`). Esse resultado ilustra a conexão entre a abordagem determinística e a abordagem de Machine Learning.

---

## Estrutura do repositório

```
fiap-fase-1-aurora-siger/
├── aurora_siger_report.ipynb     ← notebook principal (execute aqui)
├── verification_flowchart.md     ← algoritmo de decisão em Mermaid
├── telemetry_reference.md        ← valores e faixas seguras com fontes
├── requirements.txt              ← dependências com versões mínimas
└── README.md
```

---

## Como executar

**Pré-requisitos:** Python 3.9+

```bash
# Clone o repositório
git clone https://github.com/juliaramosguedes/fiap-fase-1-aurora-siger.git
cd fiap-fase-1-aurora-siger

# Instale as dependências
pip install -r requirements.txt
```

```bash
# Abra o notebook
jupyter notebook aurora_siger_report.ipynb
```

Na interface que abrir: **`Kernel → Restart & Run All`**

A primeira célula verifica automaticamente se todas as dependências estão disponíveis.

> **VS Code / Cursor:** abra `aurora_siger_report.ipynb` com a extensão Jupyter instalada e use **Run All**.

---

## Fontes dos dados

| Parâmetro | Fonte |
|---|---|
| Temperatura operacional de eletrônicos | MIT OCW — *Satellite Engineering* (Keesee, 2003) |
| Temperatura de estrutura / painéis | ESA Bulletin 87 — *Spacecraft Thermal Control* |
| Integridade estrutural (flag 0/1) | ESA Mars Express dataset — `right_flag` (Breskvar et al., 2022) |
| Energia mínima para decolagem (60%) | ESA Advanced Concepts Team (2021) |
| Pressão de tanques | NASA SBIR — *Spacecraft Thermal Management* |
| Módulos críticos | ESA ESOC — Mars Express subsystems |
| Padrão de anomalias para IA | NASA SMAP/MSL — Hundman et al., KDD 2018 |

Valores marcados como `# SIMULATED` no notebook foram criados com base em ordens de grandeza documentadas. Consulte `telemetry_reference.md` para detalhes completos.

---

## Reflexão Crítica

### Ética e responsabilidade na tomada de decisão

O algoritmo decide, com base em dados, se uma decolagem deve ou não ocorrer — uma decisão que envolve vidas e recursos irreversíveis. A escolha de arquitetura — funções puras, verificações independentes, output binário claro — não é apenas técnica: é uma escolha ética.

A análise por IA reforça essa responsabilidade. Modelos de classificação operam como "caixas pretas" — e em contextos de segurança crítica, **recall é mais importante que precision**: um falso negativo (anomalia não detectada) é mais perigoso que um abort desnecessário. O modelo não substitui o operador humano; é uma ferramenta de suporte.

A ISO 26000 (ABNT, 2010) estabelece **accountability** como princípio central de responsabilidade social: quem responde pelas decisões tomadas com base em análise de IA? Essa pergunta é real em missões espaciais.

### Impacto social da exploração espacial

A exploração espacial consome recursos energéticos e materiais intensivos. O conceito de **Triple Bottom Line** (D'Hont, 2019) — People, Planet, Profit — obriga perguntar: quem se beneficia? A que custo planetário? Qual o retorno justificável?

O Brasil gerou mais de 2 milhões de toneladas de lixo eletrônico em 2019, reciclando menos de 3% desse volume (The Global E-Waste Monitor, 2020). A cadeia produtiva de equipamentos espaciais contribui para esse fluxo.

### Sustentabilidade tecnológica

A tensão mais honesta disponível nos dados desta missão: **a mesma IA usada para otimizar o consumo de energia também o consome em volume crescente**. Data centers já respondem por ~2% do consumo elétrico global (The Green Grid, 2023), e esse percentual cresce com IA, big data e blockchain.

O PROCEL evitou 140 milhões de toneladas de CO₂ entre 1990 e 2022. O IPCC (2023) projeta que eficiência energética pode contribuir com 40% da redução de emissões até 2030.

### Consideração final

A trajetória da computação — do ábaco ao transistor, do ENIAC (150 kW) ao chip moderno — é uma narrativa de eficiência crescente com responsabilidade crescente. Construir sistemas que tomam decisões críticas exige não apenas domínio técnico, mas consciência das implicações éticas, ambientais e sociais.

Essa é a formação que a Fase 1 da FIAP propõe — e que este relatório procura demonstrar.
