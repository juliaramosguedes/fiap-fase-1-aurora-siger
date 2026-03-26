# 📡 Telemetria de Referência — Missão Aurora Siger

> Documento de fundação para o Relatório Operacional de Pré-Decolagem.
> Define os valores simulados de telemetria, faixas seguras e módulos críticos,
> com fonte declarada para cada parâmetro.
>
> Valores sem dataset específico são declarados como **simulados** — a transparência sobre o que é real e o que é ordem de grandeza é parte do design.

---

## 1. Fonte dos dados

### Dataset principal — NASA SMAP/MSL (Anomaly Detection)
- **Missão:** SMAP (Soil Moisture Active Passive) e MSL (Mars Science Laboratory / Curiosity Rover)
- **Organização:** NASA Jet Propulsion Laboratory (JPL)
- **Referência:** Hundman, K. et al. (2018). *Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding*. KDD 2018. https://github.com/khundman/telemanom
- **Kaggle:** https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl
- **Uso neste projeto:** estrutura de canais de telemetria (P = power, R = radiation, T = thermal) e padrão de anomalias anotadas como referência para a análise por IA.

### Dataset secundário — ESA Mars Express (MEX) Thermal Power
- **Missão:** Mars Express Orbiter
- **Organização:** European Space Agency (ESA) / Advanced Concepts Team
- **Referência:** Breskvar, M. et al. (2022). *Machine-learning ready data on the thermal power consumption of the Mars Express Spacecraft*. Scientific Data. https://doi.org/10.1038/s41597-022-01336-z
- **Zenodo (CSV):** https://zenodo.org/records/6417900
- **Uso neste projeto:** faixas de consumo de potência por subsistema, estrutura de módulos operacionais e variável `right_flag` (status binário 0/1) como modelo para flags de integridade.

### Referência de limites operacionais de temperatura — NASA/MIT
- **Fonte 1:** MIT OpenCourseWare — *Satellite Engineering, Spacecraft Thermal Control* (Col. John Keesee, 2003). https://ocw.mit.edu/courses/16-851-satellite-engineering-fall-2003/
- **Fonte 2:** ESA Bulletin 87 — *Current and Future Techniques for Spacecraft Thermal Control*. https://www.esa.int/esapub/bulletin/bullet87/paroli87.htm
- **Uso:** faixas de temperatura operacional por subsistema (baterias, eletrônicos digitais, propulsão).

### Referência de pressão de propulsão
- **Fonte:** NASA SBIR — *Spacecraft Thermal Management*. https://sbir.gsfc.nasa.gov/content/spacecraft-thermal-management-1
- **Uso:** faixas de pressão operacional (200 psia máximo operacional como referência de ordem de grandeza).

---

## 2. Variáveis de telemetria

### 2.1 Temperatura interna dos eletrônicos (°C)

| Parâmetro | Valor simulado |
|---|---|
| Valor atual | **22.5 °C** |
| Limite mínimo operacional | **-20 °C** |
| Limite máximo operacional | **+70 °C** |

**Fonte dos limites:** MIT OCW Satellite Engineering (2003) — eletrônicos digitais: operação entre -20°C e +70°C; ESA Bulletin 87 — "generic electronics, with an average operating range between -20 and +70°C".

---

### 2.2 Temperatura externa da estrutura (°C)

| Parâmetro | Valor simulado |
|---|---|
| Valor atual | **-45.0 °C** |
| Limite mínimo operacional | **-100 °C** |
| Limite máximo operacional | **+125 °C** |

**Fonte dos limites:** MIT OCW Satellite Engineering (2003) — estrutura/painéis solares: -100°C a +125°C (survival range). Temperatura externa negativa é nominal para uma nave em fase de decolagem interplanetária.

---

### 2.3 Integridade estrutural (flag binário)

| Parâmetro | Valor simulado |
|---|---|
| Valor atual | **1** (íntegro) |
| 0 = falha estrutural detectada | — |
| 1 = estrutura íntegra | — |

**Fonte:** ESA Mars Express dataset — variável `right_flag` (Guidance flag: 1 = yes, 0 = no), utilizada como modelo para flags booleanos de estado de subsistema. Hundman et al. (2018) — canais do tipo P (power) e status de módulos representados como valores binários normalizados.

---

### 2.4 Nível de energia das baterias (%)

| Parâmetro | Valor simulado |
|---|---|
| Valor atual | **87.3 %** |
| Capacidade total | **120.0 kWh** |
| Limite mínimo para decolagem | **60.0 %** |

**Fonte dos limites:** ESA Mars Express — missão usa 33 linhas de potência térmica; fórmula de potência disponível: `Science Power = Produced Power − Platform Power − Thermal Power`. O limite de 60% é derivado da margem de segurança operacional documentada na literatura de gestão de energia de missões espaciais (ESA Advanced Concepts Team, 2021). Valor de capacidade (120 kWh) é **simulado**, baseado na ordem de grandeza de sistemas de energia de satélites de médio porte.

---

### 2.5 Pressão dos tanques de propulsão (bar)

| Parâmetro | Valor simulado |
|---|---|
| Valor atual | **34.8 bar** |
| Limite mínimo operacional | **20.0 bar** |
| Limite máximo operacional | **40.0 bar** |

**Fonte dos limites:** NASA SBIR Spacecraft Thermal Management — "200 psia maximum expected operating pressure" (~13.8 bar) como referência de pressão de fluido de trabalho. Faixa expandida para propulsão principal com base em ordem de grandeza de motores de missão interplanetária. Os valores específicos são **simulados** com justificativa de escala física.

> [!NOTE]
> Pressão de tanques é o parâmetro com menor cobertura em datasets públicos de telemetria espacial. Os limites acima são simulados com base em ordens de grandeza documentadas, não em dados históricos de missão específica.

---

### 2.6 Status dos módulos críticos

Os módulos abaixo são derivados da estrutura de subsistemas documentada na telemetria do Mars Express (ESA) e nos canais do dataset NASA SMAP/MSL (canais P = power subsystem, R = radiation/communication).

| Módulo | Flag | Valor simulado | Referência |
|---|---|---|---|
| Propulsion System | `propulsion_ok` | `True` | Canal tipo P — NASA SMAP/MSL |
| Power Management | `power_ok` | `True` | ESA MEX — 33 power lines |
| Communication | `comms_ok` | `True` | Canal tipo R — NASA SMAP/MSL |
| Thermal Control | `thermal_ok` | `True` | ESA MEX — thermal subsystem |
| Navigation/ADCS | `navigation_ok` | `True` | ESA MEX — attitude control context data |

**Justificativa da escolha de 5 módulos:** correspondem aos subsistemas primários documentados na telemetria real do Mars Express (thermal, propulsion, power management, radio communication, attitude control) conforme ESA ESOC operations team (https://blogs.esa.int/rocketscience/2016/03/21/open-data-from-mars/).

---

## 3. Parâmetros energéticos para análise (Seção 4 do relatório)

| Parâmetro | Valor | Origem |
|---|---|---|
| Capacidade total | 120.0 kWh | Simulado — ordem de grandeza satélite médio porte |
| Carga atual | 87.3 % | Simulado — baseado em ESA MEX operação nominal |
| Consumo estimado na decolagem | 8.5 kW | Simulado — baseado em ESA MEX platform power típico |
| Fator de perda (η_perda) | 0.08 (8%) | Cap. 7 FIAP — P = I²×R; ESA MEX degradação de bateria |
| Rendimento do sistema (η) | 0.92 (92%) | Tabela 1 Cap. 7 FIAP — servidor otimizado: 80–88%; nave com sistemas novos: 92% |

**Fórmulas implementadas (Cap. 7 FIAP) — ver `compute_energy_analysis` no notebook:**
```
carga_atual_kwh  = capacidade_total_kwh × (carga_pct / 100)
energia_util     = carga_atual_kwh × rendimento
autonomia_horas  = energia_util / consumo_decolagem_kw
```

---

## 4. Decisões de modelagem

Decisões não óbvias que afetam a interpretação dos dados:

| Decisão | Escolha | Justificativa |
|---|---|---|
| Unidade de pressão | bar | Conversão de psia para SI; literatura ESA usa SI |
| Capacidade energética | 120 kWh (simulado) | Ordem de grandeza de satélite médio porte; declarado `# SIMULATED` |
| Rendimento η | 92% | Tabela 1 Cap. 7 FIAP — sistemas novos; degradação modelada pelo fator de perda |
| Fator de perda | 8% | Cap. 7 FIAP: P = I²×R aplicado como fração da carga total |
| Dataset de IA | NASA SMAP/MSL | Único dataset público com anomalias anotadas em telemetria real de nave |

---

*Missão Aurora Siger · FIAP — Ciência da Computação, 2026*

🧑‍🚀 [Julia Ramos](https://www.linkedin.com/in/juliaramosguedes) · [Carlos Eugenio](https://www.linkedin.com/in/carloseugenioandrade/) · [Matheus Fuchelberguer](https://www.linkedin.com/in/matheus-fuchelberguer-neves/) · Julio Joaquim
