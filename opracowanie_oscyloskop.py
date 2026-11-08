"""Ćw. 3 Dioda Zenera - opracowanie pomiarów metodą oscyloskopową.

Dane: oscylogramy NewFile0..3.wfm (Rigol DS1000E) dla diod 1..4,
CH1 - napięcie wejściowe U_we (generator), CH2 - napięcie wyjściowe U_wy na rezystorze R = 1 kΩ.
Wyniki (wykresy .png oraz wszystkie obliczenia w wyniki.txt i wyniki.json) trafiają
do katalogu "wyniki_oscyloskop".
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import RigolWFM.wfm as rigol

from opracowanie_punkt_po_punkcie import NAZWA

KATALOG = Path(__file__).parent
WYNIKI = KATALOG / "wyniki_oscyloskop"

PLIKI = {1: "NewFile0.wfm", 2: "NewFile1.wfm", 3: "NewFile2.wfm", 4: "NewFile3.wfm"}
R = 1000.0  # [Ω]

# Odczyty U_wy (CH2) zanotowane podczas pomiarów [V]
U_WY_ZANOTOWANE = {
    1: (-0.020, 0.680),
    2: (-0.120, 0.540),
    3: (-0.020, 0.540),
    4: (-0.020, 0.540),
}

# Rigol DS1000E, dokładność pomiaru DC: ±(3% * |odczyt| + 0,1 div + 1 mV)
OSC_PROC = 0.03
DZIALKI_POZIOME, DZIALKI_PIONOWE = 12, 8


def u_osc(U, v_div):
    """Niepewność standardowa napięcia odczytanego z oscyloskopu [V]: u = dokładność / sqrt(3)."""
    return (OSC_PROC * np.abs(U) + 0.1 * v_div + 1e-3) / np.sqrt(3)


def wczytaj_wfm(nr):
    w = rigol.Wfm.from_file(str(KATALOG / PLIKI[nr]), "1000E")
    ch1, ch2 = w.channels[0], w.channels[1]
    return dict(
        t=np.asarray(ch1.times),
        we=np.asarray(ch1.volts),
        wy=np.asarray(ch2.volts),
        v_div_1=ch1.volt_per_division,
        v_div_2=ch2.volt_per_division,
        t_div=ch1.time_scale,
        t_offset=ch1.time_offset,
        dt=ch1.seconds_per_point,
    )


def analiza(nr, o):
    U_wy_min, U_wy_max = U_WY_ZANOTOWANE[nr]
    U_we_min, U_we_max = o["we"].min(), o["we"].max()
    u_wy = lambda U: u_osc(U, o["v_div_2"])
    u_we = lambda U: u_osc(U, o["v_div_1"])

    def kierunek(U_we, U_wy):
        UD = U_we - U_wy
        return dict(
            U_we=U_we, u_U_we=u_we(U_we), U_wy=U_wy, u_U_wy=u_wy(U_wy),
            I_D=U_wy / R, u_I_D=u_wy(U_wy) / R,
            U_D=UD, u_U_D=np.hypot(u_we(U_we), u_wy(U_wy)),
        )

    return dict(
        przewodzenie=kierunek(U_we_max, U_wy_max),
        zaporowy=kierunek(U_we_min, U_wy_min),
    )


def wykres_oscylogram(nr, o, w):
    t0 = o["t_offset"]
    szer = DZIALKI_POZIOME / 2 * o["t_div"]
    m = (o["t"] >= t0 - szer) & (o["t"] <= t0 + szer)
    t_ms = o["t"][m] * 1e3

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.plot(t_ms, o["we"][m], color="goldenrod", lw=1.3, label="CH1: $U_{we}$ (generator)")
    ax.plot(t_ms, o["wy"][m], color="tab:cyan", lw=1.3, label="CH2: $U_{wy}$ na R = 1 kΩ  ($I_D = U_{wy}/R$)")
    p, z = w["przewodzenie"], w["zaporowy"]
    ax.axhline(p["U_wy"], color="tab:blue", ls="--", lw=1,
               label=f"$U_{{wy,max}}$ = {p['U_wy'] * 1e3:.0f} mV  →  $I_D$ = {p['I_D'] * 1e3:.2f} mA")
    ax.axhline(z["U_wy"], color="tab:red", ls="--", lw=1,
               label=f"$U_{{wy,min}}$ = {z['U_wy'] * 1e3:.0f} mV  →  $I_D$ = {z['I_D'] * 1e3:.2f} mA")

    ax.set_xticks(np.linspace(t0 - szer, t0 + szer, DZIALKI_POZIOME + 1) * 1e3)
    v = o["v_div_2"]
    ax.set_yticks(np.arange(-DZIALKI_PIONOWE / 2, DZIALKI_PIONOWE / 2 + 1) * v)
    ax.set_xlim((t0 - szer) * 1e3, (t0 + szer) * 1e3)
    ax.set_ylim(-DZIALKI_PIONOWE / 2 * v, DZIALKI_PIONOWE / 2 * v)
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
    ax.grid(alpha=0.5)
    ax.set_xlabel("t [ms]")
    ax.set_ylabel("U [V]")
    ax.set_title(f"{NAZWA[nr]} - oscylogram  (CH1, CH2: {o['v_div_1'] * 1e3:.0f} mV/div; "
                 f"podstawa czasu: {o['t_div'] * 1e6:.0f} µs/div)", fontsize=10.5)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(WYNIKI / f"dioda{nr}_oscylogram.png", dpi=200)
    plt.close(fig)


def zaokr(x, n=6):
    return float(f"{x:.{n}g}")


def wyniki_json(osc, wyn):
    o1 = osc[1]
    wynik = {
        "metoda": "oscyloskopowa",
        "zalozenia": {
            "oscyloskop": "Rigol DS1000E",
            "CH1": "U_we - napięcie z generatora (wejście X makiety)",
            "CH2": "U_wy - napięcie na rezystorze R (wyjście Y makiety)",
            "R_Ohm": R,
            "czulosc_CH1_V_na_div": o1["v_div_1"],
            "czulosc_CH2_V_na_div": o1["v_div_2"],
            "podstawa_czasu_s_na_div": o1["t_div"],
            "okres_probkowania_s": o1["dt"],
            "dokladnosc_U": "±(3% * |U| + 0,1 div + 1 mV)",
            "niepewnosc_standardowa": "u = dokładność / sqrt(3)",
            "wzory": "I_D = U_wy / R,  U_D = U_we - U_wy,  u(U_D) = sqrt(u(U_we)^2 + u(U_wy)^2),  u(I_D) = u(U_wy) / R",
            "U_wy": "zanotowane U_min i U_max kanału CH2",
            "U_we": "minimum i maksimum kanału CH1 z pliku .wfm",
        },
        "diody": {},
    }
    for nr, w in wyn.items():
        dioda = {"nazwa": NAZWA[nr], "plik": PLIKI[nr]}
        for kier in ("przewodzenie", "zaporowy"):
            k = w[kier]
            dioda[kier] = {
                "U_we_V": zaokr(k["U_we"]), "u_U_we_V": zaokr(k["u_U_we"]),
                "U_wy_V": zaokr(k["U_wy"]), "u_U_wy_V": zaokr(k["u_U_wy"]),
                "I_D_mA": zaokr(k["I_D"] * 1e3), "u_I_D_mA": zaokr(k["u_I_D"] * 1e3),
                "U_D_V": zaokr(k["U_D"]), "u_U_D_V": zaokr(k["u_U_D"]),
            }
        wynik["diody"][str(nr)] = dioda
    return wynik


def wyniki_txt(osc, wyn):
    o1 = osc[1]
    linie = [
        "OPRACOWANIE - METODA OSCYLOSKOPOWA",
        "=" * 78,
        "",
        "Założenia:",
        "  oscyloskop: Rigol DS1000E;  CH1 = U_we (generator),  CH2 = U_wy (na R = 1 kΩ)",
        f"  czułość CH1 i CH2: {o1['v_div_1'] * 1e3:.0f} mV/div;  podstawa czasu: {o1['t_div'] * 1e6:.0f} µs/div;"
        f"  próbkowanie co {o1['dt'] * 1e6:.0f} µs",
        "  dokładność U: ±(3% * |U| + 0,1 div + 1 mV);  u = dokładność / sqrt(3)",
        "  I_D = U_wy / R,  U_D = U_we - U_wy",
        "  u(I_D) = u(U_wy) / R,  u(U_D) = sqrt(u(U_we)^2 + u(U_wy)^2)",
        "  U_wy - zanotowane U_min i U_max (CH2);  U_we - minimum i maksimum CH1 z pliku .wfm",
        "",
        "PODSUMOWANIE",
        "-" * 78,
        f"{'dioda':<24}{'kierunek':<14}{'U_we [V]':>15}{'U_wy [V]':>15}{'I_D [mA]':>16}{'U_D [V]':>15}",
    ]
    for nr, w in wyn.items():
        for kier in ("przewodzenie", "zaporowy"):
            k = w[kier]
            linie.append(
                f"{NAZWA[nr] if kier == 'przewodzenie' else '':<24}{kier:<14}"
                f"{k['U_we']:>8.2f} ± {k['u_U_we']:.2f}{k['U_wy']:>8.2f} ± {k['u_U_wy']:.2f}"
                f"{k['I_D'] * 1e3:>9.2f} ± {k['u_I_D'] * 1e3:.2f}{k['U_D']:>8.2f} ± {k['u_U_D']:.2f}")
    linie.append("")
    return "\n".join(linie)


def main():
    WYNIKI.mkdir(exist_ok=True)
    for stary in [*WYNIKI.glob("*.png"), *WYNIKI.glob("*.txt"), *WYNIKI.glob("*.json")]:
        stary.unlink()

    osc, wyn = {}, {}
    for nr in PLIKI:
        osc[nr] = wczytaj_wfm(nr)
        wyn[nr] = analiza(nr, osc[nr])
        wykres_oscylogram(nr, osc[nr], wyn[nr])

    tekst = wyniki_txt(osc, wyn)
    (WYNIKI / "wyniki.txt").write_text(tekst, encoding="utf-8")
    (WYNIKI / "wyniki.json").write_text(
        json.dumps(wyniki_json(osc, wyn), ensure_ascii=False, indent=2), encoding="utf-8")
    print(tekst)


if __name__ == "__main__":
    main()
