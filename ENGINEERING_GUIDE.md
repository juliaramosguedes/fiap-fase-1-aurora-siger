# Guia de Engenharia: Aurora Siger Report

> Um walkthrough do `aurora_siger_report.ipynb` — do panorama geral até as linhas individuais de código.
> Destinado a engenheiros júnior lendo esta base de código pela primeira vez.

---

## Índice

1. [Panorama Geral — O que esse notebook faz](#1-panorama-geral--o-que-esse-notebook-faz)
2. [Arquitetura do Código — Como está organizado](#2-arquitetura-do-código--como-está-organizado)
3. [Modelo de Execução — A ordem das células importa](#3-modelo-de-execução--a-ordem-das-células-importa)
4. [Seção 1 — Telemetria](#4-seção-1--telemetria)
5. [Seção 2 — Verificação de Pré-Lançamento](#5-seção-2--verificação-de-pré-lançamento)
6. [Seção 3 — Análise Energética](#6-seção-3--análise-energética)
7. [Seção 4 — Análise Assistida por IA](#7-seção-4--análise-assistida-por-ia)
8. [Padrões Recorrentes](#8-padrões-recorrentes)
9. [Mapa de Fluxo de Dados](#9-mapa-de-fluxo-de-dados)
10. [Glossário](#10-glossário)

---

## 1. Panorama Geral — O que esse notebook faz

**Em uma frase:** Este notebook simula um checklist de pré-lançamento de uma espaçonave para a missão fictícia Aurora Siger, combinando verificação manual, cálculos energéticos e dois modelos de machine learning.

**A história de ponta a ponta:**

```
Leituras simuladas de sensores (telemetria)
        ↓
Verificações manuais baseadas em regras (temperatura, pressão, energia, módulos)
        ↓
Cálculos energéticos (autonomia em horas)
        ↓
Modelos de IA treinados em dados sintéticos:
  - IsolationForest → detecção de anomalias não supervisionada
  - DecisionTree    → classificação binária supervisionada
        ↓
Resumo de risco + decisão final de lançamento
```

**Saída final (quando tudo está nominal):**
```
>>> PRONTO PARA DECOLAR <<<
O universo espera. Boa viagem, Aurora Siger. 🖖
```

---

## 2. Arquitetura do Código — Como está organizado

O notebook segue uma arquitetura em camadas. Cada camada depende apenas das camadas abaixo dela.

```
Camada 4 — Exibição      print_telemetry(), print_checklist(), print_energy_analysis(), ...
                                │
Camada 3 — Computação    compute_energy_analysis(), train_isolation_forest(), train_decision_tree(), ...
                                │
Camada 2 — Tipos de dado TelemetryReading, EnergyAnalysis, IsolationForestResult, DecisionTreeResult
                                │
Camada 1 — Constantes    RANDOM_STATE, SEPARATOR, valores de limite (TEMP_INTERNAL_MIN, ...)
```

**Por que isso importa:** As constantes são definidas uma vez e reutilizadas em todo lugar — nas funções de verificação *e* na geração do dataset. Mude um limite em um lugar, e tanto a verificação manual quanto os dados de treinamento da IA se atualizam automaticamente.

---

## 3. Modelo de Execução — A ordem das células importa

### Célula 1 — Badge do Colab (markdown)
Apenas um badge com link para o Google Colab. Sem código.

### Célula 2 — Título e Visão Geral (markdown)
O cabeçalho da missão. Apresenta as quatro seções e nota que valores marcados com `# SIMULATED` são estimados por ordem de grandeza.

### Célula 3 — Imports e Constantes Globais

**Esta deve ser executada primeiro.** Todos os imports são declarados aqui para evitar surpresas de `NameError` se as células rodarem fora de ordem.

```python
from __future__ import annotations    # permite type hints como List['NomeDaClasse'] no Python 3.9-
from dataclasses import dataclass      # decorator para containers de dados limpos (ver Seção 4 → Padrões)
from typing import Tuple, Dict, List   # auxiliares para anotações de tipo

import numpy as np                     # arrays numéricos e geração de números aleatórios
import pandas as pd                    # dados tabulares (usado no dataset de IA)
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    recall_score, precision_score, f1_score,
)
```

**Constantes globais definidas aqui:**

| Constante | Valor | Propósito |
|---|---|---|
| `RANDOM_STATE` | `42` | Semente fixa — garante que todo modelo e divisão seja reproduzível entre execuções |
| `REPORT_WIDTH` | `60` | Largura em caracteres para todos os separadores visuais |
| `SEPARATOR` | `"=" * 60` | A linha horizontal impressa entre seções |

```python
np.random.seed(RANDOM_STATE)  # define a semente aleatória global do NumPy uma vez
```

**`format_status(flag: bool) -> str`**
Um helper minúsculo que converte um booleano em string legível para humanos.
```python
format_status(True)  # → "OK"
format_status(False) # → "FALHA"
```
Usado em todas as funções de exibição para não duplicar lógica de display.

### Célula 4 — Verificação de Dependências

Lê `requirements.txt`, tenta importar cada pacote e imprime um resumo.

```python
def _parse_requirements(path: str = "requirements.txt") -> list:
    # Abre o requirements.txt
    # Remove especificadores de versão (>=, ==, ~=) com regex
    # Retorna apenas os nomes dos pacotes como lista
    ...

missing = [inst for inst, imp in check_pairs
           if importlib.util.find_spec(imp) is None]
```

`importlib.util.find_spec(name)` retorna `None` se um pacote não está instalado — sem precisar importar para verificar.

O dict `_INSTALL_TO_IMPORT` lida com pacotes onde o nome de instalação difere do nome de importação:
```python
{"scikit-learn": "sklearn", "jupyter": None}
# scikit-learn é instalado como "scikit-learn" mas importado como "sklearn"
# jupyter não tem módulo importável para verificar, então pula
```

---

## 4. Seção 1 — Telemetria

**Alto nível:** Define quais leituras de sensores temos e armazena os valores atuais da missão Aurora Siger.

### O dataclass `TelemetryReading`

```python
@dataclass(frozen=True)
class TelemetryReading:
    internal_temperature: float   # temperatura do compartimento de eletrônica (°C)
    external_temperature: float   # temperatura da casca/estrutura (°C)
    structural_integrity: int     # 1 = intacto, 0 = falha
    energy_level_pct: float       # carga da bateria (%)
    tank_pressure_bar: float      # pressão do tanque de propulsão (bar)
    propulsion_ok: bool
    power_ok: bool
    comms_ok: bool
    thermal_ok: bool
    navigation_ok: bool
```

**O que é `@dataclass`?**
Um decorator do Python que gera automaticamente `__init__`, `__repr__` e `__eq__` para uma classe. Em vez de escrever um construtor manualmente, você apenas lista os campos com seus tipos.

**O que significa `frozen=True`?**
O objeto se torna imutável — você não pode alterar seus campos após a criação. Isso é intencional: uma vez que você lê um sensor, aquela leitura não deve mudar no meio do cálculo. Se você tentar `TELEMETRY.energy_level_pct = 50`, o Python lança `FrozenInstanceError`.

**A instância `TELEMETRY` — valores atuais da missão:**

```python
TELEMETRY = TelemetryReading(
    internal_temperature=22.5,   # dentro de -20°C a +70°C → nominal
    external_temperature=-45.0,  # dentro de -100°C a +125°C → nominal
    structural_integrity=1,      # intacto
    energy_level_pct=87.3,       # acima do mínimo de 60% → nominal
    tank_pressure_bar=34.8,      # dentro de 20-40 bar → nominal
    propulsion_ok=True,
    power_ok=True,
    comms_ok=True,
    thermal_ok=True,
    navigation_ok=True,
)
```

**Constantes do sistema de energia (usadas na Seção 3):**

```python
TOTAL_CAPACITY_KWH    = 120.0  # capacidade total da bateria
LAUNCH_CONSUMPTION_KW =   8.5  # potência de pico consumida no lançamento
SYSTEM_EFFICIENCY     =  0.92  # 92% da energia é realmente utilizável (η)
LOSS_FACTOR           =  0.08  # 8% de perdas resistivas (P = I²×R)
```

### `print_telemetry(reading: TelemetryReading) -> None`

Função apenas de exibição. Recebe um `TelemetryReading` e imprime uma tabela formatada. Não retorna nada (`None`).

Note o uso de formatação com f-string:
```python
f"  Temperatura interna   : {reading.internal_temperature:>8.1f} °C"
#                                                           ↑
# :>8.1f = alinhar à direita em 8 caracteres, 1 casa decimal, formato float
```
Isso mantém todos os números alinhados na saída.

---

## 5. Seção 2 — Verificação de Pré-Lançamento

**Alto nível:** Define limites de segurança e verifica se cada valor de telemetria está dentro dos limites aceitáveis.

### Limites de Segurança — Fonte Única da Verdade

```python
TEMP_INTERNAL_MIN: float = -20.0
TEMP_INTERNAL_MAX: float =  70.0
TEMP_EXTERNAL_MIN: float = -100.0
TEMP_EXTERNAL_MAX: float =  125.0
ENERGY_MIN_PCT:    float =  60.0
PRESSURE_MIN_BAR:  float =  20.0
PRESSURE_MAX_BAR:  float =  40.0
AUTONOMY_MIN_HOURS: float = 2.0
```

Essas constantes aparecem aqui **e** são usadas na Seção 4 para gerar o dataset de treinamento da IA. Isso significa que as regras do que é "seguro" são definidas exatamente uma vez — a IA aprende da mesma definição usada nas verificações manuais.

### Strings de Decisão

```python
DECISION_READY:   str = "PRONTO PARA DECOLAR"
DECISION_ABORTED: str = "DECOLAGEM ABORTADA"
```

Definidas como constantes para que nunca ocorram erros de digitação em outro lugar. Qualquer comparação como `if decision == DECISION_READY` sempre estará correta.

### As Funções de Verificação — Funções Puras

Cada verificação é uma **função pura**: recebe entrada, retorna um booleano e não tem efeitos colaterais (não imprime, não muda nenhum estado global).

```python
def check_internal_temperature(value: float) -> bool:
    return TEMP_INTERNAL_MIN <= value <= TEMP_INTERNAL_MAX
    # Python permite comparações encadeadas: a <= b <= c é válido
```

```python
def check_critical_modules(reading: TelemetryReading) -> bool:
    return all([
        reading.propulsion_ok,
        reading.power_ok,
        reading.comms_ok,
        reading.thermal_ok,
        reading.navigation_ok,
    ])
    # all([True, True, False]) → False
    # Falha na verificação inteira se mesmo um subsistema estiver down
```

### `run_checks(reading, autonomy_hours) -> Dict[str, bool]`

Chama todas as funções de verificação e retorna um dicionário:

```python
{
    "internal_temperature": True,
    "external_temperature": True,
    "structural_integrity": True,
    "energy_level": True,
    "tank_pressure": True,
    "critical_modules": True,
    "autonomy": True,
}
```

Sem impressão aqui — apenas dados brutos.

### `decide_launch(checks: Dict[str, bool]) -> str`

```python
def decide_launch(checks: Dict[str, bool]) -> str:
    return DECISION_READY if all(checks.values()) else DECISION_ABORTED
    # all({...}.values()) → True apenas se cada valor no dict for True
```

### `verify_launch_readiness(reading, autonomy_hours) -> Tuple[str, Dict[str, bool]]`

A função de alto nível que conecta a Seção 2. Chama `run_checks` e `decide_launch`, retorna ambos.

```python
decision, checks = verify_launch_readiness(TELEMETRY, energy.autonomy_hours)
# decision → "PRONTO PARA DECOLAR" ou "DECOLAGEM ABORTADA"
# checks   → o dict de resultados individuais
```

**Desempacotamento de tupla:** O Python permite atribuir múltiplos valores de retorno de uma vez — `a, b = funcao()` funciona quando a função retorna `(a, b)`.

### `print_checklist(decision, checks) -> None`

Formata e imprime o checklist com ícones:
```python
icon   = "✔︎" if passed else "💥"
result = "OK" if passed else "FALHA"
label  = CHECK_LABELS.get(name, name.replace("_", " ").title())
# CHECK_LABELS mapeia chave interna → label de exibição em português
# .get(chave, padrão) retorna o label se encontrado, ou formata a chave como fallback
```

---

## 6. Seção 3 — Análise Energética

**Alto nível:** Calcula quanta energia utilizável está disponível e por quantas horas a espaçonave pode operar — a verificação de "autonomia" que alimenta de volta a Seção 2.

### O dataclass `EnergyAnalysis`

```python
@dataclass(frozen=True)
class EnergyAnalysis:
    total_capacity_kwh:   float  # capacidade máxima da bateria
    charge_pct:           float  # carga atual como percentual
    current_charge_kwh:   float  # charge_pct convertido para kWh
    energy_losses_kwh:    float  # energia perdida em calor (I²×R)
    efficiency:           float  # fração da energia que é utilizável
    energy_available_kwh: float  # o que está realmente disponível após perdas
    consumption_kw:       float  # quanto de potência o lançamento usa por hora
    autonomy_hours:       float  # quantas horas de operação são possíveis
    load_factor:          float  # razão: carga_atual / capacidade_total
```

### `compute_energy_analysis(...) -> EnergyAnalysis`

Todas as fórmulas do Capítulo 7 da FIAP:

```python
# Passo 1: Quanta energia está armazenada atualmente?
current_charge_kwh = total_capacity_kwh * (charge_pct / 100)
# Exemplo: 120 kWh × (87.3 / 100) = 104.76 kWh

# Passo 2: Perdas resistivas — P = I²×R da lei de Ohm
energy_losses_kwh = current_charge_kwh * loss_factor
# Exemplo: 104.76 × 0.08 = 8.38 kWh desperdiçados como calor

# Passo 3: Energia utilizável — η (eta) = E_útil / E_total
energy_available_kwh = current_charge_kwh * efficiency
# Exemplo: 104.76 × 0.92 = 96.38 kWh disponíveis

# Passo 4: Por quanto tempo podemos operar com essa taxa de consumo?
autonomy_hours = energy_available_kwh / consumption_kw
# Exemplo: 96.38 kWh ÷ 8.5 kW = 11.34 horas

# Passo 5: Fator de carga — estamos usando maior parte da bateria?
load_factor = current_charge_kwh / total_capacity_kwh
# Exemplo: 104.76 / 120.0 = 0.87 (87% carregado)
```

**Por que retornar um dataclass em vez de variáveis separadas?**
Retornar todos os resultados em um objeto facilita passar adiante, registrar ou comparar sem precisar de muitos argumentos de função.

### A Chamada de Computação

```python
energy = compute_energy_analysis(
    total_capacity_kwh=TOTAL_CAPACITY_KWH,
    charge_pct=TELEMETRY.energy_level_pct,  # conecta Seção 1 → Seção 3
    consumption_kw=LAUNCH_CONSUMPTION_KW,
    efficiency=SYSTEM_EFFICIENCY,
    loss_factor=LOSS_FACTOR,
)

decision, checks = verify_launch_readiness(TELEMETRY, energy.autonomy_hours)
# energy.autonomy_hours alimenta de volta a função check_autonomy()
```

---

## 7. Seção 4 — Análise Assistida por IA

**Alto nível:** Treina dois modelos de machine learning em um dataset sintético para verificar independentemente se a leitura de telemetria atual parece anômala.

### Por que Dois Modelos?

| Modelo | Tipo | Precisa de rótulos? | Ponto forte |
|---|---|---|---|
| `IsolationForest` | Não supervisionado | Não | Detecta novidades sem rótulos históricos |
| `DecisionTree` | Supervisionado | Sim | Aprende regras explícitas; interpretável |

Ambos os modelos classificam a mesma leitura de telemetria e suas saídas são comparadas no resumo final de risco.

### Geração do Dataset

**Nomes das features (5 colunas, espelhando os campos de `TelemetryReading`):**
```python
FEATURE_NAMES = [
    "internal_temp", "external_temp", "structural_integrity",
    "energy_pct", "tank_pressure",
]
```

**Faixas nominais** ficam ligeiramente dentro dos limiares de segurança (buffer de 5-10 unidades):
```python
NOMINAL_RANGES = {
    "internal_temp": (TEMP_INTERNAL_MIN + 5, TEMP_INTERNAL_MAX - 5),  # -15 a +65
    ...
}
```

**Faixas de anomalia** ficam deliberadamente fora dos limites seguros:
```python
ANOMALY_RANGES = {
    "internal_temp": [
        (TEMP_INTERNAL_MIN - 30, TEMP_INTERNAL_MIN - 1),  # muito frio
        (TEMP_INTERNAL_MAX + 1,  TEMP_INTERNAL_MAX + 40), # muito quente
    ],
    ...
}
```

**`generate_nominal_missions(n: int) -> np.ndarray`**
Gera `n` linhas onde cada feature é amostrada de sua faixa nominal.

**`generate_anomalous_missions(n: int) -> np.ndarray`**
Gera `n` linhas onde uma feature por linha é substituída por um valor anômalo. Todas as outras features permanecem nominais — simulando uma falha de ponto único.

```python
def _inject_fault(row: np.ndarray, fault_feature: str) -> np.ndarray:
    col    = FEATURE_NAMES.index(fault_feature)  # encontra o índice da coluna
    result = row.copy()                           # não muta o original
    result[col] = _sample_anomaly(fault_feature) # substitui por valor ruim
    return result
```

**`build_dataset(n_nominal, n_anomalous)`**
Combina os dois arrays, cria rótulos e retorna um DataFrame do Pandas:
```python
X = np.vstack([X_nominal, X_anomalous])  # empilha verticalmente: 1500 + 500 = 2000 linhas
y = [0, 0, ..., 1, 1, ...]               # 0 = nominal, 1 = anomalia
```

### 4A — IsolationForest (Não Supervisionado)

**O que faz:** Isola outliers dividindo aleatoriamente o espaço de features. Pontos que precisam de menos divisões para serem isolados são considerados anomalias.

**Conceitos-chave:**

`StandardScaler` — transforma features para que cada uma tenha média=0 e desvio padrão=1. O IsolationForest é sensível à escala, então normalizar evita que qualquer feature única domine.

```python
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)   # aprende média/std dos dados, depois transforma
```

`contamination` — diz ao modelo qual fração dos dados esperar como anômala:
```python
CONTAMINATION_RATIO = N_ANOMALOUS / (N_NOMINAL + N_ANOMALOUS)  # 500/2000 = 0.25
```

**Remapeamento da saída:**
O IsolationForest usa sua própria convenção (`-1` para anomalia, `+1` para nominal). O código mapeia isso para a convenção do projeto (`1` para anomalia, `0` para nominal):
```python
labels = (preds == IF_ANOMALY_LABEL).astype(int)
# preds == -1 cria um array booleano: True onde é anomalia
# .astype(int) converte True → 1, False → 0
```

**`classify_with_isolation_forest(result, reading) -> str`**
Recebe um modelo treinado e um único `TelemetryReading`, escala as features usando o *mesmo scaler* treinado no dataset completo, e retorna a string de classificação.

```python
features = telemetry_to_feature_array(reading)  # TelemetryReading → array numpy 2D
scaled   = result.scaler.transform(features)    # aplica scaling (não re-treina!)
pred     = result.model.predict(scaled)[0]
```

### 4B — DecisionTreeClassifier (Supervisionado)

**O que faz:** Aprende uma série de regras if-then a partir de exemplos rotulados. A árvore divide os dados pela feature que melhor separa nominal de anômalo.

**Escolhas de configuração:**

`class_weight='balanced'` — há 3× mais amostras nominais do que anômalas. Esse peso compensa para que o modelo não ignore a classe minoritária.

`StratifiedKFold(n_splits=5)` — validação cruzada que preserva a proporção de classes em cada fold. Mais confiável do que uma única divisão treino/teste.

`cross_val_score(scoring="recall")` — **recall é a métrica prioritária** aqui. Em sistemas de segurança crítica, perder uma anomalia real (falso negativo) é pior do que um alarme falso (falso positivo).

**Validação cruzada explicada:**
```
Dataset completo (2000 amostras)
   ├── Fold 1: [teste] [treino][treino][treino][treino]
   ├── Fold 2: [treino][teste] [treino][treino][treino]
   ├── Fold 3: [treino][treino][teste] [treino][treino]
   ├── Fold 4: [treino][treino][treino][teste] [treino]
   └── Fold 5: [treino][treino][treino][treino][teste]

Recall médio em 5 execuções = estimativa mais confiável do que uma única divisão
```

**As regras aprendidas (primeiros 3 níveis da árvore):**
```
|--- structural_integrity <= 0.50
|   |--- classe: 1          ← anomalia (falha estrutural detectada imediatamente)
|--- structural_integrity >  0.50
|   |--- energy_pct <= 61.48
|   |   |--- classe: 1      ← anomalia (energia baixa)
|   |--- energy_pct >  61.48
|   |   |--- tank_pressure <= 19.55
```

A árvore descobriu independentemente as mesmas regras das verificações manuais — começando pela feature mais decisiva (`structural_integrity`), depois `energy_pct`, depois `tank_pressure`.

### 4C — Resumo de Risco da IA

`classify_risk_level(anomaly_probability: float) -> str`

```python
if anomaly_probability < 0.20:  → "BAIXO"
if anomaly_probability < 0.50:  → "MODERADO"
else:                           → "ALTO"
```

`build_ai_risk_summary(...)` — uma **função pura** que constrói o resumo completo de risco como string. Ela não imprime nada — quem chama faz `print(build_ai_risk_summary(...))`.

Verifica se ambos os modelos concordam:
```python
consensus = isolation_forest_result == dt_classification
# Se IsolationForest diz NOMINAL mas DecisionTree diz ANOMALIA DETECTADA, isso é um sinal de alerta
```

### Célula 16 — Relatório Final

`print_final_report(...)` — coleta resultados das Seções 3 e 4 e imprime o resumo da missão.

---

## 8. Padrões Recorrentes

### Padrão 1: `@dataclass(frozen=True)`

Usado em: `TelemetryReading`, `EnergyAnalysis`, `IsolationForestResult`, `DecisionTreeResult`

**Por quê:** Containers imutáveis que agrupam valores relacionados. Sem risco de mutar acidentalmente resultados no meio do cálculo. O flag `frozen=True` é o equivalente em Python de "somente leitura após criação."

### Padrão 2: Funções Puras

A maioria das funções neste notebook são **puras** — recebem entradas, calculam algo e retornam um valor. Sem mudança de estado global, sem impressão.

As funções `print_*` são as únicas exceções, e são explicitamente rotuladas como "Display-only, no return value" em seus docstrings.

**Por que essa separação importa:**
- Você pode testar `compute_energy_analysis()` de forma independente
- Você pode trocar o formato de exibição sem tocar na lógica
- Você pode chamar `verify_launch_readiness()` de outro código sem efeitos colaterais indesejados

### Padrão 3: Constantes como Fonte Única da Verdade

```python
ENERGY_MIN_PCT: float = 60.0

# Usado na Seção 2 (verificação manual):
def check_energy_level(pct: float) -> bool:
    return pct >= ENERGY_MIN_PCT

# Usado na Seção 4 (geração de dataset para IA):
NOMINAL_RANGES = {
    "energy_pct": (ENERGY_MIN_PCT + 2, 100.0),
    ...
}
```

Mudar `ENERGY_MIN_PCT` uma vez atualiza tanto a verificação manual quanto os dados de treinamento da IA.

### Padrão 4: Calcular depois Exibir

Cada seção segue o mesmo padrão em duas etapas:
1. Chamar uma função de computação → obter resultado como dataclass
2. Passar o resultado para uma função `print_*` → exibi-lo

```python
energy = compute_energy_analysis(...)  # etapa 1: calcular
print_energy_analysis(energy)          # etapa 2: exibir
```

### Padrão 5: Anotações de Tipo

```python
def run_checks(
    reading: TelemetryReading,
    autonomy_hours: float,
) -> Dict[str, bool]:
```

Anotações de tipo não impõem tipos em tempo de execução no Python — são documentação para humanos e ferramentas de análise estática. Elas dizem exatamente o que uma função espera e retorna.

---

## 9. Mapa de Fluxo de Dados

```
Célula 3: Imports + Constantes
        │
        ▼
Célula 4: Verificação de dependências (lê requirements.txt)
        │
        ▼
Célula 6: TELEMETRY = TelemetryReading(...)        ← Seção 1
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
Célula 10: compute_energy_analysis(...)   Célula 8: Limiares de segurança definidos
           → EnergyAnalysis                         funções check_*() definidas
           │                                        │
           └──────────────────────────────┘
                          │
                          ▼
                 verify_launch_readiness(TELEMETRY, energy.autonomy_hours)
                          │
                          ├── decision ("PRONTO..." ou "ABORTADA")
                          └── checks (Dict[str, bool])
                          │
                          ▼
Célula 12: build_dataset(1500, 500)         ← Seção 4 (usa mesmas constantes da Seção 2)
           → X (features), y (rótulos), df
           │
           ├── Célula 13: train_isolation_forest(X, y, ...) → IsolationForestResult
           │                                                   │
           │   classify_with_isolation_forest(iso, TELEMETRY) → "NOMINAL"
           │
           └── Célula 14: train_decision_tree(X, y, TELEMETRY, ...) → DecisionTreeResult
                                                                        │
Célula 15: build_ai_risk_summary(iso_result, iso.recall, dt_result) → str
          │
          ▼
Célula 16: print_final_report(energy, iso_result, dt_result, decision)
```

---

## 10. Glossário

| Termo | Significado em linguagem simples |
|---|---|
| `dataclass` | Uma classe Python que gera automaticamente código repetitivo como `__init__` a partir dos campos declarados |
| `frozen=True` | Torna um dataclass imutável — os campos não podem mudar após a criação |
| Função pura | Uma função sem efeitos colaterais — a mesma entrada sempre dá a mesma saída |
| `RANDOM_STATE = 42` | Uma semente fixa para que operações aleatórias sejam reproduzíveis entre execuções |
| `np.random.seed()` | Define a semente global de números aleatórios do NumPy |
| `StandardScaler` | Transforma features para terem média=0 e desvio padrão=1, para que nenhuma feature domine pela escala |
| `contamination` | Fração dos dados que o IsolationForest espera ser anômala |
| `class_weight='balanced'` | Compensa o desequilíbrio de classes dando mais peso à classe minoritária |
| `StratifiedKFold` | Validação cruzada que preserva a proporção de classes em cada fold |
| Recall | De todas as anomalias reais, quantas o modelo detectou? (métrica prioritária em sistemas de segurança) |
| Precisão | De tudo que o modelo sinalizou como anomalia, quantos eram realmente anomalias? |
| F1 score | Média harmônica de recall e precisão |
| Matriz de confusão | Tabela mostrando verdadeiros positivos, falsos positivos, verdadeiros negativos, falsos negativos |
| Autonomia (horas) | Por quanto tempo a espaçonave pode operar com a energia disponível na taxa de consumo atual |
| Fator de carga (FC) | Razão entre a carga de energia atual e a capacidade total |
| η (eta) | Eficiência do sistema — fração da energia armazenada que é utilizável após as perdas |
| `# SIMULATED` | Valores estimados por ordem de grandeza, documentados em `telemetry.md` |
