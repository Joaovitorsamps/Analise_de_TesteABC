# Célula 1 - Imports e caminhos
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import os
import IPython.display as display

# Caminho da pasta com os CSVs
caminho_base = r"C:\Users\joaov\Desktop\teste_tecnico\CSV"
f_clientes = os.path.join(caminho_base, "clientes.csv")
f_pedidos  = os.path.join(caminho_base, "pedidos.csv")

# Opcional: caso queira alterar os padrões de "confirmado"
CONFIRM_PATTERNS = [
    "%confirm%",   # confirma, confirmed, confirmada...
    "%paid%",      # paid
    "%pago%",      # pago
]
CONFIRM_EQUIV = ("1","true","sim","yes","ok")
#Leitura dos CSVs e normalização de colunas
df_clientes = pd.read_csv(f_clientes, encoding="latin-1", dtype=str, low_memory=False)
df_pedidos  = pd.read_csv(f_pedidos,  encoding="latin-1", dtype=str, low_memory=False)

# Normalizar colunas
df_clientes.columns = df_clientes.columns.str.lower().str.strip()
df_pedidos.columns  = df_pedidos.columns.str.lower().str.strip()

# Tenta normalizar o nome da chave do cliente para 'cliente_id'
def rename_if_exists(df, possible_names, target):
    for name in possible_names:
        if name in df.columns:
            df.rename(columns={name: target}, inplace=True)
            return True
    return False

rename_if_exists(df_clientes, ["cliente_id"], "cliente_id")
rename_if_exists(df_pedidos,  ["cliente_id"], "cliente_id")

required_clientes = {"cliente_id","grupo","estado"}
required_pedidos  = {"cliente_id","status"}

missing_clientes = required_clientes - set(df_clientes.columns)
missing_pedidos  = required_pedidos  - set(df_pedidos.columns)

if missing_clientes:
    raise ValueError(f"Faltam colunas em clientes.csv: {missing_clientes}. Renomeie-as ou ajuste o CSV.")
if missing_pedidos:
    raise ValueError(f"Faltam colunas em pedidos.csv: {missing_pedidos}. Renomeie-as ou ajuste o CSV.")

#limpar espaços nas strings das colunas
df_clientes["cliente_id"] = df_clientes["cliente_id"].astype(str).str.strip()
df_pedidos["cliente_id"]  = df_pedidos["cliente_id"].astype(str).str.strip()
df_clientes["grupo"] = df_clientes["grupo"].astype(str).str.strip().str.upper()
df_clientes["estado"] = df_clientes["estado"].astype(str).str.strip()
df_pedidos["status"] = df_pedidos["status"].astype(str).str.strip()
# Célula 3 - Registrar os DataFrames no DuckDB e montar + executar a query SQL
conn = duckdb.connect(database=":memory:")

# Registra os DataFrames como tabelas no DuckDB
conn.register("clientes", df_clientes)
conn.register("pedidos", df_pedidos)

# Monta a cláusula SQL para detectar "confirmados" (padrões + equivalentes)
# Usamos LOWER(status) para comparação case-insensitive.
confirm_where_clauses = []
for p in CONFIRM_PATTERNS:
    # DuckDB usa LIKE com % para pattern; já temos % nos padrões
    confirm_where_clauses.append(f"lower(p.status) LIKE lower('{p}')")

equals_clause = " OR ".join([f"lower(p.status) = '{val}'" for val in CONFIRM_EQUIV])

where_confirm = " OR ".join(confirm_where_clauses + ([equals_clause] if equals_clause else []))

sql = f"""
SELECT
  c.estado AS estado,
  c.grupo  AS grupo,
  COUNT(*) AS compras_confirmadas
FROM pedidos p
JOIN clientes c
  ON p.cliente_id = c.cliente_id
WHERE ({where_confirm})
GROUP BY c.estado, c.grupo
ORDER BY c.estado, c.grupo;
"""

df_agg = conn.execute(sql).df()
df_agg
#SQL Cancelamentos por estado - Análise de Pareto
sql_pareto_cancelamentos_por_estado = """
WITH counts AS (
  SELECT
    c.estado,
    COUNT(*) AS cnt
  FROM pedidos p
  JOIN clientes c ON p.cliente_id = c.cliente_id
  WHERE lower(p.status) LIKE '%cancelado%'
  GROUP BY c.estado
  ORDER BY cnt DESC
),
tot AS (
  SELECT SUM(cnt) AS total FROM counts
)
SELECT
  estado,
  cnt,
  SUM(cnt) OVER (ORDER BY cnt DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative,
  100.0 * SUM(cnt) OVER (ORDER BY cnt DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) / (SELECT total FROM tot) AS cumulative_pct
FROM counts
ORDER BY cnt DESC;
"""
df_pareto = conn.execute(sql_pareto_cancelamentos_por_estado).df()
df_pareto
sql_teste_serie_temporal = """
WITH parsed AS (
  SELECT
    DATE_TRUNC('day', STRPTIME(p.data_pedido, '%d/%m/%Y %H:%M'))::DATE AS dia,
    c.grupo AS grupo,
    COUNT(*) AS cnt
  FROM pedidos p
  JOIN clientes c ON p.cliente_id = c.cliente_id
  WHERE lower(c.grupo) IN ('a','b','c')
  GROUP BY dia, c.grupo
)
SELECT dia, grupo, cnt
FROM parsed
ORDER BY dia, grupo;
"""
df_time = conn.execute(sql_teste_serie_temporal).df()
df_time.head(10)
#Pivot de serie temporal dos grupos A, B, C
df_time_pivot = df_time.pivot(index="dia", columns="grupo", values="cnt").fillna(0)

cols_order = [c for c in ["A","B","C"] if c in df_time_pivot.columns]
other = [c for c in df_time_pivot.columns if c not in cols_order]
df_time_pivot = df_time_pivot[cols_order + other]
display(df_time_pivot.head())
# Pivot para formato wide (colunas A, B, C lado a lado) e tratamento
df_pivot = df_agg.pivot(index="estado", columns="grupo", values="compras_confirmadas").fillna(0)

expected = ["A","B","C"]
cols_in_order = [g for g in expected if g in df_pivot.columns]
other_groups = [c for c in df_pivot.columns if c not in cols_in_order]
final_cols = cols_in_order + other_groups
if final_cols:
    df_pivot = df_pivot[final_cols]

display(df_pivot)
# Célula — Dashboard: monta os 3 gráficos juntos (Testes por Estado, Pareto Cancelamentos, Testes em Função do Tempo)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---------- Parâmetros de layout ----------
fig = plt.figure(constrained_layout=True, figsize=(16, 12))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.1])

ax1 = fig.add_subplot(gs[0, 0])  
ax2 = fig.add_subplot(gs[0, 1])   
ax3 = fig.add_subplot(gs[1, :])   

# ---------- Gráfico 1: Testes por Estado (barras agrupadas) ----------
states = df_pivot.index.tolist()
groups = df_pivot.columns.tolist()
n_states = len(states)
n_groups = len(groups)

# Ajustáveis:
spacing = 1.8        
group_width = 0.8
width = group_width / max(1, n_groups)

x = np.arange(n_states) * spacing

for i, grp in enumerate(groups):
    heights = df_pivot[grp].values
    ax1.bar(x + i*width, heights, width=width, label=str(grp))

ax1.set_xticks(x + group_width/2)
ax1.set_xticklabels(states, rotation=45, ha="right")
ax1.set_xlabel("Estado")
ax1.set_ylabel("Quantidade de compras confirmadas")
ax1.set_title("Testes por Estado")   # título solicitado
ax1.legend(title="Grupo")
ax1.grid(axis='y', linestyle='--', alpha=0.3)

# ---------- Gráfico 2: Pareto de Cancelamentos ----------
# df_pareto tem: estado, cnt, cumulative, cumulative_pct
if not df_pareto.empty:
    estados_p = df_pareto['estado'].astype(str).tolist()
    counts_p = df_pareto['cnt'].astype(float).tolist()
    cum_pct = df_pareto['cumulative_pct'].astype(float).tolist()

    ax2.bar(estados_p, counts_p)
    ax2.set_xlabel("Estado")
    ax2.set_ylabel("Cancelamentos (contagem)")

    ax2_twin = ax2.twinx()
    ax2_twin.plot(estados_p, cum_pct, color='C1', marker='o', linestyle='-')
    ax2_twin.set_ylabel("Cumulative %")
    ax2_twin.set_ylim(0, 100)

    ax2.set_title("Cancelamentos por estados")
    # Anotações opcionais: mostrar % cumulativo sobre as barras
    for i, v in enumerate(counts_p):
        ax2.text(i, v + max(counts_p)*0.01, str(int(v)), ha='center', va='bottom', fontsize=8)
    for i, p in enumerate(cum_pct):
        ax2_twin.text(i, p + 2, f"{p:.1f}%", ha='center', va='bottom', fontsize=8, color='C1')
else:
    ax2.text(0.5, 0.5, "Nenhum cancelamento encontrado", ha='center', va='center')
    ax2.set_title("Cancelamentos por estados")
    ax2.set_xticks([])

# ---------- Gráfico 3: Série temporal (Testes em Função do Tempo) ----------
# df_time_pivot index: dia (datetime.date), col: A/B/C
if not df_time_pivot.empty:
    for col in df_time_pivot.columns:
        ax3.plot(df_time_pivot.index, df_time_pivot[col], marker='o', label=str(col))
    ax3.set_xlabel("Dia")
    ax3.set_ylabel("Número de pedidos")
    ax3.set_title("Testes em Função do Tempo")
    ax3.legend(title="Grupo")
    ax3.grid(alpha=0.3)
else:
    ax3.text(0.5, 0.5, "Sem dados de pedidos para série temporal", ha='center', va='center')
    ax3.set_title("Testes em Função do Tempo")

plt.suptitle("Dashboard: Testes e Cancelamentos", fontsize=16, y=0.98)
plt.show()
