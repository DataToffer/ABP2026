import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
df = pd.read_csv("ascenso_albacete_2015_2025_v2.csv")

# Calcular estadísticas para equipos ascendidos
ascendidos = df[df["Ascendido"] == 1].copy()
ascendidos["PPG"] = ascendidos["Pts"] / 42
ascendidos["GF/PG"] = ascendidos["GF"] / 42
ascendidos["GA/PG"] = ascendidos["GA"] / 42
ascendidos["DG"] = ascendidos["GF"] - ascendidos["GA"]

stats_summary = ascendidos[["Pts", "GF", "GA", "DG", "PPG", "GF/PG", "GA/PG"]].describe()

# Rendimiento del Albacete
albacete = df[df["Equipo"] == "Albacete"].copy()
albacete["PPG"] = albacete["Pts"] / 42
albacete["GF/PG"] = albacete["GF"] / 42
albacete["GA/PG"] = albacete["GA"] / 42
albacete["DG"] = albacete["GF"] - albacete["GA"]

# Brecha Albacete 2024-25 vs Media de Ascenso
media_ascenso = ascendidos.mean(numeric_only=True)
alba_2025 = albacete[albacete["Temporada"] == "2024-25"].iloc[0]
brecha = {
    "Pts": media_ascenso["Pts"] - alba_2025["Pts"],
    "GF": media_ascenso["GF"] - alba_2025["GF"],
    "GA": alba_2025["GA"] - media_ascenso["GA"]
}

# Escenario óptimo
escenario = {
    "Pts": alba_2025["Pts"] + 20,
    "GF": alba_2025["GF"] + 15,
    "GA": alba_2025["GA"] - 15,
}
escenario["DG"] = escenario["GF"] - escenario["GA"]
escenario["PPG"] = escenario["Pts"] / 42

# Validación estadística
z_score = (escenario["PPG"] - media_ascenso["PPG"]) / ascendidos["PPG"].std()

# Test de normalidad Shapiro-Wilk
shapiro_test = stats.shapiro(ascendidos["PPG"])

# Regresión múltiple (Pts ~ GF + GA)
X = ascendidos[["GF", "GA"]]
X = sm.add_constant(X)
y = ascendidos["Pts"]
model = sm.OLS(y, X).fit()

# Predicción con escenario
escenario_pred = model.predict([1, escenario["GF"], escenario["GA"]])[0]

# Guardar archivo resumen
with open("resultado_analisis.txt", "w") as f:
    f.write("Resumen Estadístico Ascendidos\n")
    f.write(str(stats_summary))
    f.write("\n\nBrecha Albacete vs Media Ascenso:\n")
    f.write(str(brecha))
    f.write("\n\nEscenario Proyectado:\n")
    f.write(str(escenario))
    f.write("\n\nZ-Score Escenario: {:.2f}\n".format(z_score))
    f.write("Shapiro-Wilk W={:.3f}, p-value={:.3f}\n".format(*shapiro_test))
    f.write("\nModelo de Regresión:\n")
    f.write(str(model.summary()))
    f.write("\n\nPredicción de puntos para escenario proyectado: {:.2f}\n".format(escenario_pred))
