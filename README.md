
# ⚽ Ascenso Albacete 2026 · Análisis Predictivo con Python 🧠📊

Este repositorio contiene el análisis estadístico completo del rendimiento del **Albacete Balompié** 🦇 en Segunda División (LaLiga Hypermotion) durante las últimas temporadas, con el objetivo de evaluar sus opciones reales de ascenso a Primera División en la temporada 2025–26.

Incluye:
- 📈 Cálculos estadísticos detallados (media, desviación típica, z-scores, percentiles).
- 📊 Simulación de escenarios óptimos de ascenso.
- 🧪 Validación estadística del modelo.
- 🐍 Script completo en Python para replicar todos los análisis.

---

## 📂 Estructura del repositorio

```
📁 ascenso-albacete-2026/
├── ascenso_albacete_2015_2025_v2.csv        # Dataset principal con 23 temporadas
├── ascenso_albacete_analysis.py             # Script Python con todos los cálculos
├── Ascenso Albacete 2026.pdf                # Informe profesional con resultados
├── gráficos/                                # Gráficos generados con matplotlib
└── README.md                                # Este archivo
```

---

## 🚀 Requisitos

Instala las librerías necesarias con:

```bash
pip install pandas matplotlib seaborn scipy numpy
```

---

## ▶️ Cómo ejecutar

1. Descarga el repositorio.
2. Asegúrate de tener el archivo `ascenso_albacete_2015_2025_v2.csv` en la misma carpeta que el script.
3. Ejecuta el análisis con:

```bash
python ascenso_albacete_analysis.py
```

El script mostrará en consola los resultados clave y generará los gráficos automáticamente.

---

## 📊 Resultados destacados

- El análisis estima que con **78 puntos**, el Albacete tendría un ~80% de probabilidad de ascenso directo.
- Las proyecciones estadísticas se han validado a partir de los últimos **20 equipos ascendidos**.
- El modelo emplea distribución normal para validar el perfil de ascenso en base al PPG (puntos por partido), diferencia de goles y eficiencia defensiva/ofensiva.

---

## 📌 Referencias

- Datos extraídos de fuentes oficiales y plataformas de fútbol.
- Validación estadística basada en z-scores y test de normalidad.

---

## 📣 Autor

El proyecto ha sido desarrollado como parte de una iniciativa de análisis avanzado aplicada al deporte y comunicación de datos.  
🧠 Creado por Juanfer Sánchez · [LinkedIn](https://www.linkedin.com/in/juanfersanchez)

---

¡Aúpa Alba! 🏁🔴⚪
