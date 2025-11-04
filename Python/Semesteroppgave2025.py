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
from matplotlib.patches import Patch

# --- Konfigurasjon / tilstand ---
# Fargepalett for nedbørskategorier (lav -> høy)
COLOR_PALETTE = ['orange', 'gray', 'blue', 'darkblue', 'black']

# Bakgrunnsbilde for kartet - legg bildet i Python-mappen og oppdater filnavnet her
# Støtter: PNG, JPG, JPEG (prøver både små og store bokstaver)
BACKGROUND_IMAGE_NAME = 'StorBergen2.png'  # Endre til ditt bildefilnavn her

# Bakgrunnsfarger for skjemaet
FIGURE_BG_COLOR = '#25433c'      # Mørkegrønn bakgrunnsfarge for hele vinduet
GRAPH_BG_COLOR = '#25433c'       # Mørkegrønn bakgrunnsfarge for nedbørsgrafen (venstre side)
MAP_BG_COLOR = '#f6f8fb'         # Bakgrunnsfarge for kartet (hvis ingen bilde) - lys grå

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
    axGraph.set_facecolor(GRAPH_BG_COLOR)  # Sett bakgrunnsfarge på nytt
    draw_the_map()
    axMap.set_title(f"C: ({x:.1f},{y:.1f}) - click rød er estimert", color='white', fontsize=12, fontweight='bold')


    axMap.text(x, y, s=label_from_nedbor(aarsnedbor), color='white', fontsize=10, ha='center', va='center')
    global last_pred_monthly
    last_pred_monthly = y_pred

    title_prefix = 'Nedbør per måned' if display_mode == 'Måned' else 'Nedbør per kvartal'
    axGraph.set_title(f"{title_prefix}, Årsnedbør {int(aarsnedbor)} mm", color='white', fontsize=12, fontweight='bold')

    if display_mode == 'Måned':
        colorsPred = [color_from_nedbor(v * 12) for v in y_pred]
        axGraph.bar(months, y_pred, color=colorsPred)
        mean_value = np.mean(y_pred)
        print(f"Gjennomsnitt (mm per måned): {mean_value:.2f}")
        axGraph.axhline(
            y=mean_value,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f"Gj.snitt {mean_value:.0f} mm/mnd")
        axGraph.legend()
    else:
        quarters = aggregate_to_quarters(y_pred)
        colorsPred = [color_from_nedbor(v * 4) for v in quarters]
        axGraph.bar(np.arange(1, 5), quarters, color=colorsPred)

    axMap.scatter(x, y, c=color_from_nedbor(aarsnedbor), s=size_from_nedbor(aarsnedbor) * 3.5, marker="o")
    axMap.scatter(x, y, c="red", s=size_from_nedbor(aarsnedbor)*2.5, marker="o")
    draw_label_and_ticks()

    add_legend_left()
    plt.draw()

def draw_label_and_ticks() -> None:
    """Oppdater x-akse etiketter og område etter valgt visningsmodus."""
    if display_mode == 'Måned':
        xlabels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
        axGraph.set_xticks(np.linspace(1, 12, 12))
        axGraph.set_xticklabels(xlabels, color='white')
        axGraph.set_xlim(0.5, 12.5)
    else:
        xlabels = ['Q1', 'Q2', 'Q3', 'Q4']
        axGraph.set_xticks(np.linspace(1, 4, 4))
        axGraph.set_xticklabels(xlabels, color='white')
        axGraph.set_xlim(0.5, 4.5)
    
    # Sett farger på akser og labels for synlighet på mørk bakgrunn
    axGraph.tick_params(colors='white', labelsize=9)
    axGraph.spines['bottom'].set_color('white')
    axGraph.spines['left'].set_color('white')
    axGraph.spines['top'].set_color('white')
    axGraph.spines['right'].set_color('white')
    axGraph.yaxis.label.set_color('white')
    axGraph.xaxis.label.set_color('white')


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
fig = plt.figure(figsize=(10, 4), facecolor=FIGURE_BG_COLOR)
axGraph = fig.add_axes((0.07, 0.05, 0.3, 0.80))
axGraph.set_facecolor(GRAPH_BG_COLOR)
axMap = fig.add_axes((0.39, 0.05, 0.54, 0.85))
axMode = fig.add_axes((0.935, 0.50, 0.03, 0.25))
axMode.set_facecolor(FIGURE_BG_COLOR)  # Sett bakgrunnsfarge på radiobutton-området
axMode.set_xticks([])  # Fjern x-akse
axMode.set_yticks([])  # Fjern y-akse
for spine in axMode.spines.values():
    spine.set_visible(False)  # Fjern kanter
draw_label_and_ticks()
# Last bakgrunnsbilde - prøver flere varianter av filnavn (små/store bokstaver, ulike filformater)
img = None
base_name = Path(BACKGROUND_IMAGE_NAME).stem  # Navn uten filendelse
extensions = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG']
for ext in extensions:
    candidate = base_name + ext
    img_path = Path(__file__).with_name(candidate)
    if img_path.exists():
        try:
            img = mpimg.imread(img_path)
            print(f"Lastet bakgrunnsbilde: {candidate}")
            break
        except Exception as e:
            print(f"Kunne ikke laste {candidate}: {e}")
            continue

if img is not None:
    axMap.imshow(img, extent=(0, 12, 0, 10))
else:
    # No image available; draw background color instead
    axMap.set_facecolor(MAP_BG_COLOR)
    axMap.set_xlim(0, 12)
    axMap.set_ylim(0, 10)
axMap.set_title("Årsnedbør Stor Bergen", color='white', fontsize=12, fontweight='bold')
axGraph.set_title("Per måned", color='white', fontsize=12, fontweight='bold')
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

legend_labels = [
    '< 1300 mm',
    '1300–1699 mm',
    '1700–2499 mm',
    '2500–3199 mm',
    '≥ 3200 mm'
]
legend_handles = [Patch(color=c, label=l) for c, l in zip(COLOR_PALETTE, legend_labels)]

def add_legend_left():
    leg = axGraph.legend(handles=legend_handles, loc='upper left', frameon=True, fancybox=True)
    leg.set_title('Årsnedbør (mm)')
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_alpha(0.85)

draw_the_map()
add_legend_left()

plt.connect('button_press_event', on_click)

# Radio-knapper for å bytte mellom Måned og Kvartal
radio = RadioButtons(axMode, ("Måned", "Kvartal"), activecolor='#66ff66')  # Lys grønn farge for aktiv knapp
radio.set_active(0)

# Styling av radiobuttons - sett bakgrunn og farger
# RadioButtons bruker patches (sirkler) som kan aksesseres via patches
for patch in axMode.patches:
    patch.set_edgecolor('white')  # Hvit kant
    patch.set_facecolor('#25433c')  # Mørkegrønn bakgrunn (samme som skjemaet)
    patch.set_linewidth(2.5)

# Styling av labels (tekst)
for label in radio.labels:
    label.set_color('white')  # Hvit tekst
    label.set_fontsize(11)
    label.set_fontweight('bold')

def on_mode_change(label):
    global display_mode
    display_mode = label
    axGraph.cla()
    axGraph.set_facecolor(GRAPH_BG_COLOR)  # Sett bakgrunnsfarge på nytt
    draw_label_and_ticks()
    
    # Visuell feedback: oppdater radiobutton styling
    for patch in axMode.patches:
        patch.set_edgecolor('white')
        patch.set_linewidth(2.5)
    if last_pred_monthly is not None:
        title_prefix = 'Nedbør per måned' if display_mode == 'Måned' else 'Nedbør per kvartal'
        aarsnedbor = float(np.sum(last_pred_monthly))
        axGraph.set_title(f"{title_prefix}, Årsnedbør {int(aarsnedbor)} mm", color='white', fontsize=12, fontweight='bold')
        if display_mode == 'Måned':
            colorsPred = [color_from_nedbor(v * 12) for v in last_pred_monthly]
            axGraph.bar(np.arange(1, 13), last_pred_monthly, color=colorsPred)
            mean_value = np.mean(last_pred_monthly)
            print(f"Gjennomsnitt (mm per måned): {mean_value:.2f}")
            axGraph.axhline(
                y=mean_value,
                color='red',
                linestyle='--',
                linewidth=2,
                label=f"Gj.snitt {mean_value:.0f} mm/mnd")
            axGraph.legend()
        else:
            quarters = aggregate_to_quarters(last_pred_monthly)
            colorsPred = [color_from_nedbor(v * 4) for v in quarters]
            axGraph.bar(np.arange(1, 5), quarters, color=colorsPred)
    add_legend_left()
    plt.draw()

radio.on_clicked(on_mode_change)
plt.show()


