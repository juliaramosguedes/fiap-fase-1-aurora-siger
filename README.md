# 🚀 Missão Aurora Siger

## Relatório Operacional de Pré-Decolagem

*Atividade Integradora — Fase 1 | Ciência da Computação — FIAP*

**Julia Ramos · Carlos Eugenio · Julio Guma · Matheus Fuchens**

![Python 3.9](https://img.shields.io/badge/Python-3.9+-blue) ![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange) ![Scikit-Learn 1.x](https://img.shields.io/badge/Scikit--Learn-1.x-green)

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

- **IsolationForest** (não supervisionado): aprende o padrão nominal e detecta qualquer desvio sem precisar de anomalias históricas rotuladas. Em operações espaciais, anomalias rotuladas são raras — esse modelo é mais robusto nesse cenário. Atingiu **recall de 87,5%**.
- **LogisticRegression** (supervisionado): classifica cada leitura como nominal ou anomalia com base em histórico sintético gerado a partir dos mesmos limiares de segurança. Serve como segundo parecer e permite quantificar a probabilidade de anomalia.

Usar os dois e comparar o consenso é mais confiável do que depender de um único modelo. Em sistemas de segurança crítica, **recall é a métrica prioritária**: um falso negativo — anomalia não detectada — é mais perigoso que um abort desnecessário.

---

## Decisões técnicas

**Functional programming** — todas as funções de verificação são puras: sem estado interno, sem efeitos colaterais, mesmo input sempre produz mesmo output. Isso torna cada função testável individualmente, o que é mandatório em sistemas de segurança crítica.

**Imutabilidade** — `TelemetryReading` e `EnergyAnalysis` são dataclasses frozen. Representam uma leitura num momento específico do tempo (T-0). Não faz sentido modificar dados de telemetria após a leitura.

**Single source of truth** — os limiares de segurança são definidos uma vez na Seção 3 e reutilizados no dataset de treinamento da IA. Se o limite de energia mínima muda de 60% para 65%, muda em um lugar e propaga para o sistema inteiro.

**Dados de referência reais** — os limiares operacionais vêm de documentação técnica da NASA e ESA. Valores sem dataset de origem são marcados como `# SIMULATED` no código e documentados em `telemetry_reference.md`.

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
    LogisticRegression (supervised): NOMINAL
    Model consensus               : YES

  Anomaly identification
    Anomaly probability           : 18.2%
    IsolationForest recall        : 87.5%
    LogisticRegression recall     : 30.0%

  Risk suggestion
    Risk level                    : LOW
    Recommended action            : PROCEED WITH CAUTION
============================================================
```

---

## Estrutura do repositório

```
fiap-fase-1-aurora-siger/
├── aurora_siger_report.ipynb     ← notebook principal (execute aqui)
├── verification_flowchart.md     ← algoritmo de decisão
├── telemetry_reference.md        ← valores e faixas seguras com fontes
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
pip install notebook numpy pandas scikit-learn
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
