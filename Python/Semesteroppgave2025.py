# importing modules and packages
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import preprocessing
import matplotlib.image as mpimg
from sklearn.preprocessing import PolynomialFeatures
from matplotlib.widgets import RadioButtons
from matplotlib.backend_bases import MouseEvent
from typing import Iterable

# --- Konfigurasjon / tilstand ---
# Fargepalett for nedbørskategorier (lav -> høy)
COLOR_PALETTE = ['orange', 'gray', 'blue', 'darkblue', 'black']

display_mode = 'Måned'  # or 'Kvartal'
last_pred_monthly = None  # stores last 12-length prediction vector

def draw_the_map() -> None:
    """Tegn bakgrunnskartet og punktene for årlig nedbør per posisjon."""
    axMap.cla()
    if img is not None:
        axMap.imshow(img, extent=(0, 13, 0, 10))
    df_year = df.groupby(['X', 'Y']).agg({'Nedbor': 'sum'}).reset_index()
    xr = df_year['X'].tolist()
    yr = df_year['Y'].tolist()
    nedborAar = df_year['Nedbor']
    ColorList = [color_from_nedbor(n) for n in nedborAar]
    axMap.scatter(xr, yr, c=ColorList, s=size_from_nedbor(nedborAar/12), alpha=1)
    labels = [label_from_nedbor(n) for n in nedborAar]
    for i, y in enumerate(xr):
        axMap.text(xr[i], yr[i], s=labels[i], color='white', fontsize=10, ha='center', va='center')

def index_from_nedbor(x: float) -> int:
    if x < 1300: return 0
    if x < 1700: return 1
    if x < 2500: return 2
    if x < 3200: return 3
    return 4

def color_from_nedbor(nedbor: float) -> str:
    return COLOR_PALETTE[index_from_nedbor(nedbor)]
def size_from_nedbor(nedbor: float) -> int:
    return 350
def label_from_nedbor(nedbor: float) -> str:
    return str(int(nedbor / 100))

def on_click(event: MouseEvent) -> None:
    """Klikk i kartet: estimer nedbør for valgt posisjon og tegn grafen."""
    global marked_point
    if event.inaxes != axMap:
        return

    marked_point = (event.xdata, event.ydata)
    x,y = marked_point

    vectors = []
    months = np.linspace(1,12,12)
    for mnd in months:
        vectors.append([x,y,mnd])
    AtPoint = np.vstack(vectors)
    # fitting the model, and predict for each month
    AtPointM = poly.fit_transform(AtPoint)
    y_pred = model.predict(AtPointM)
    aarsnedbor = sum(y_pred)
    axGraph.cla()
    draw_the_map()
    axMap.set_title(f"C: ({x:.1f},{y:.1f}) - click rød er estimert")


    axMap.text(x, y, s=label_from_nedbor(aarsnedbor), color='white', fontsize=10, ha='center', va='center')
    global last_pred_monthly
    last_pred_monthly = y_pred

    title_prefix = 'Nedbør per måned' if display_mode == 'Måned' else 'Nedbør per kvartal'
    axGraph.set_title(f"{title_prefix}, Årsnedbør {int(aarsnedbor)} mm")

    if display_mode == 'Måned':
        colorsPred = [color_from_nedbor(v * 12) for v in y_pred]
        axGraph.bar(months, y_pred, color=colorsPred)
    else:
        quarters = aggregate_to_quarters(y_pred)
        colorsPred = [color_from_nedbor(v * 4) for v in quarters]
        axGraph.bar(np.arange(1, 5), quarters, color=colorsPred)

    axMap.scatter(x, y, c=color_from_nedbor(aarsnedbor), s=size_from_nedbor(aarsnedbor) * 3.5, marker="o")
    axMap.scatter(x, y, c="red", s=size_from_nedbor(aarsnedbor)*2.5, marker="o")
    draw_label_and_ticks()
    plt.draw()

def draw_label_and_ticks() -> None:
    """Oppdater x-akse etiketter og område etter valgt visningsmodus."""
    if display_mode == 'Måned':
        xlabels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
        axGraph.set_xticks(np.linspace(1, 12, 12))
        axGraph.set_xticklabels(xlabels)
        axGraph.set_xlim(0.5, 12.5)
    else:
        xlabels = ['Q1', 'Q2', 'Q3', 'Q4']
        axGraph.set_xticks(np.linspace(1, 4, 4))
        axGraph.set_xticklabels(xlabels)
        axGraph.set_xlim(0.5, 4.5)


def aggregate_to_quarters(monthly_values: Iterable[float]) -> np.ndarray:
    """Summer 12 månedsverdier til 4 kvartalsverdier (Q1..Q4)."""
    mv = np.asarray(monthly_values)
    return np.array([
        mv[0:3].sum(),
        mv[3:6].sum(),
        mv[6:9].sum(),
        mv[9:12].sum()
    ])

# Create the figures
fig = plt.figure(figsize=(10, 4))
axGraph = fig.add_axes((0.05, 0.07, 0.35, 0.85))
axMap = fig.add_axes((0.41, 0.07, 0.54, 0.85))
axMode = fig.add_axes((0.96, 0.50, 0.03, 0.25))
draw_label_and_ticks()
# Last bakgrunnsbilde ved behov (PNG/PNG variant); fall tilbake til lys bakgrunn
img = None
for candidate in ['StorBergen2.png', 'StorBergen2.PNG']:
    img_path = Path(__file__).with_name(candidate)
    if img_path.exists():
        img = mpimg.imread(img_path)
        break

if img is not None:
    axMap.imshow(img, extent=(0, 13, 0, 10))
else:
    # No image available; draw a light background grid instead
    axMap.set_facecolor('#f6f8fb')
    axMap.set_xlim(0, 13)
    axMap.set_ylim(0, 10)
axMap.set_title("Årsnedbør Stor Bergen")
axGraph.set_title("Per måned")
axMap.axis('off')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0) # Adjust the figure to fit the image
axMap.margins(x=0.01, y=0.01)  # Adjust x and y margins

# Les nedbørsdata relativt til denne filen, og del i tren/test-sett
csv_path = Path(__file__).with_name('NedborX.csv')
df = pd.read_csv(csv_path)
marked_point = (0,0)
ns = df['Nedbor']
X = df.drop('Nedbor',  axis=1)
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(
    X_poly, ns, test_size=0.25)

# Tren en enkel lineær regresjonsmodell (polynomial features)
model = LinearRegression()
model.fit(X_train, Y_train) # fitting the model
Y_pred = model.predict(X_test)

# Check model quality
r_squared = r2_score(Y_test, Y_pred)
print(f"R-squared: {r_squared:.2f}")
print('mean_absolute_error (mnd) : ', mean_absolute_error(Y_test, Y_pred))

draw_the_map()

plt.connect('button_press_event', on_click)

# Radio-knapper for å bytte mellom Måned og Kvartal
radio = RadioButtons(axMode, ("Måned", "Kvartal"))
radio.set_active(0)

def on_mode_change(label):
    global display_mode
    display_mode = label
    axGraph.cla()
    draw_label_and_ticks()
    if last_pred_monthly is not None:
        title_prefix = 'Nedbør per måned' if display_mode == 'Måned' else 'Nedbør per kvartal'
        aarsnedbor = float(np.sum(last_pred_monthly))
        axGraph.set_title(f"{title_prefix}, Årsnedbør {int(aarsnedbor)} mm")
        if display_mode == 'Måned':
            colorsPred = [color_from_nedbor(v * 12) for v in last_pred_monthly]
            axGraph.bar(np.arange(1, 13), last_pred_monthly, color=colorsPred)
        else:
            quarters = aggregate_to_quarters(last_pred_monthly)
            colorsPred = [color_from_nedbor(v * 4) for v in quarters]
            axGraph.bar(np.arange(1, 5), quarters, color=colorsPred)
    plt.draw()

radio.on_clicked(on_mode_change)
plt.show()


