"""Ćw. 3 Dioda Zenera - opracowanie pomiarów metodą "punkt po punkcie".

Dane: pliki "Dioda N.tsv" (kolumny: U [V], I [A]) zapisane z multimetrów BRYMEN BM811.
Wyniki (wykresy .png oraz wszystkie obliczenia w wyniki.txt i wyniki.json) trafiają
do katalogu "wyniki_punkt_po_punkcie".
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

KATALOG = Path(__file__).parent
WYNIKI = KATALOG / "wyniki_punkt_po_punkcie"

PLIKI = {
    1: "Dioda 1 .tsv",
    2: "Dioda 2.tsv",
    3: "Dioda 3.tsv",
    4: "Dioda 4.tsv",
}
DIODY_ZENERA = (2, 3, 4)
NAZWA = {1: "Dioda 1 (prostownicza)", 2: "Dioda 2 (Zenera)", 3: "Dioda 3 (Zenera)", 4: "Dioda 4 (Zenera)"}
KOLOR = {1: "tab:blue", 2: "tab:orange", 3: "tab:green", 4: "tab:red"}

# Zakresy prądu [mA] użyte do dopasowania prostych
I_MIN_FIT_PRZEWODZENIE = 30.0  # kierunek przewodzenia: I >= 30 mA  -> V_bi
I_MIN_FIT_ZAPOROWY = 20.0  # kierunek zaporowy: |I| >= 20 mA     -> R_Z, U_ZK
I_ZT = 20.0  # |I| punktu pracy, w którym odczytujemy U_Z [mA]
RS_ZAKRES = (5.0, 50.0)  # przedział |I| dla rezystancji statycznej [mA]
RS_LICZBA_PUNKTOW = 10

# BRYMEN BM811 (5000 działek), DC: U: ±(0,08% + 2 c), I (zakres 50 mA): ±(0,2% + 4 c)
U_PROC, U_CYFRY = 0.0008, 2
I_PROC, I_CYFRY, I_ROZDZ = 0.002, 4, 1e-5  # rozdzielczość zakresu 50 mA: 0,01 mA


def rozdzielczosc_U(U):
    """Rozdzielczość odczytu napięcia: 1 mV na zakresie 5 V, 10 mV na zakresie 50 V."""
    U = np.abs(np.asarray(U, dtype=float))
    ma_cyfre_mV = np.abs(np.round(U * 1000) % 10) > 0.5
    return np.where((U < 5.0) | ma_cyfre_mV, 1e-3, 1e-2)


def u_U(U):
    """Niepewność standardowa napięcia [V]: u = dokładność / sqrt(3)."""
    return (U_PROC * np.abs(U) + U_CYFRY * rozdzielczosc_U(U)) / np.sqrt(3)


def u_I(I):
    """Niepewność standardowa prądu [A]: u = dokładność / sqrt(3)."""
    return (I_PROC * np.abs(I) + I_CYFRY * I_ROZDZ) / np.sqrt(3)


def wczytaj(nr):
    """Wczytuje dane i usuwa powtórzone odczyty (rejestracja ciągła multimetru)."""
    d = pd.read_csv(KATALOG / PLIKI[nr], sep=r"\s+", header=None, names=["U", "I"])
    d = d.drop_duplicates().reset_index(drop=True)
    d["uU"] = u_U(d["U"])
    d["uI"] = u_I(d["I"])
    return d


def dopasuj(U, I_mA):
    """Regresja liniowa I = a*U + b (I w mA, U w V); zwraca a, b, Δa, Δb, R^2."""
    r = linregress(U, I_mA)
    return r.slope, r.intercept, r.stderr, r.intercept_stderr, r.rvalue**2


def najblizszy(wartosci, cel):
    return int(np.argmin(np.abs(np.asarray(wartosci) - cel)))


def analiza_przewodzenia(d):
    p = d[(d["U"] > 0) & (d["I"] > 0)]
    fit = p[p["I"] * 1e3 >= I_MIN_FIT_PRZEWODZENIE]
    a, b, da, db, r2 = dopasuj(fit["U"], fit["I"] * 1e3)
    Vbi = -b / a
    uVbi = np.sqrt((b / a**2 * da) ** 2 + (db / a) ** 2)
    return dict(a=a, b=b, da=da, db=db, r2=r2, n=len(fit), Vbi=Vbi, uVbi=uVbi)


def analiza_zaporowa(d):
    z = d[(d["U"] < 0) & (d["I"] < 0)].sort_values("I", ascending=False).reset_index(drop=True)
    I_mA = z["I"].to_numpy() * 1e3
    U = z["U"].to_numpy()

    fit = I_mA <= -I_MIN_FIT_ZAPOROWY
    a, b, da, db, r2 = dopasuj(U[fit], I_mA[fit])
    RZ = 1e3 / a  # [Ω], bo a w mA/V
    uRZ = 1e3 * da / a**2
    U0 = -b / a  # przecięcie prostej z osią napięcia

    # Napięcie progowe: punkt pomiarowy najbliższy przecięciu prostej z osią U;
    # prąd w tym punkcie to minimalny prąd stabilizacji I_min.
    k = najblizszy(U, U0)
    UZK, Imin = U[k], z["I"].iloc[k]

    # Napięcie Zenera: punkt pomiarowy najbliższy |I| = I_ZT
    m = najblizszy(I_mA, -I_ZT)
    UZ, IZ = U[m], z["I"].iloc[m]

    # Rezystancja statyczna w RS_LICZBA_PUNKTOW punktach równomiernie rozłożonych w |I|
    I_max_pom = -I_mA.min()
    cele = np.linspace(RS_ZAKRES[0], min(RS_ZAKRES[1], I_max_pom), RS_LICZBA_PUNKTOW)
    idx = [najblizszy(I_mA, -c) for c in cele]
    rs = z.iloc[idx].copy()
    rs["RS"] = rs["U"] / rs["I"]  # [Ω]
    rs["uRS"] = np.sqrt((rs["uU"] / rs["I"]) ** 2 + (rs["U"] * rs["uI"] / rs["I"] ** 2) ** 2)

    return dict(
        a=a, b=b, da=da, db=db, r2=r2, n=int(fit.sum()), RZ=RZ, uRZ=uRZ, U0=U0,
        UZK=UZK, uUZK=float(u_U(UZK)), Imin=Imin, uImin=float(u_I(Imin)),
        UZ=UZ, uUZ=float(u_U(UZ)), IZ=IZ, uIZ=float(u_I(IZ)), rs=rs,
    )


def errorbar(ax, d, **kw):
    ax.errorbar(d["U"], d["I"] * 1e3, xerr=d["uU"], yerr=d["uI"] * 1e3,
                fmt="o", ms=2.2, elinewidth=0.6, capsize=1.2, **kw)


def wykres_pelny(nr, d, prz, zap):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    errorbar(ax, d, color=KOLOR[nr], ecolor="0.35", label="pomiar")
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6)

    Ux = np.linspace(prz["Vbi"] - 0.004, d["U"].max() + 0.005, 50)
    ax.plot(Ux, prz["a"] * Ux + prz["b"], "k--", lw=1.2,
            label=f"prosta dla I ≥ {I_MIN_FIT_PRZEWODZENIE:.0f} mA ($R^2$ = {prz['r2']:.4f})")
    ax.plot(prz["Vbi"], 0, "kD", ms=6, label=f"$V_{{bi}}$ = ({prz['Vbi']:.3f} ± {prz['uVbi']:.3f}) V")

    if zap:
        Ux = np.linspace(d["U"].min() - 0.03, zap["U0"] + 0.05, 50)
        ax.plot(Ux, zap["a"] * Ux + zap["b"], color="0.2", ls=(0, (5, 2, 1, 2)), lw=1.2,
                label=f"prosta dla |I| ≥ {I_MIN_FIT_ZAPOROWY:.0f} mA ($R^2$ = {zap['r2']:.4f}), "
                      f"$R_Z$ = ({zap['RZ']:.2f} ± {zap['uRZ']:.2f}) Ω")
        ax.axhline(zap["Imin"] * 1e3, color="tab:purple", ls=":", lw=1.2,
                   label=f"$I_{{min}}$ = ({zap['Imin'] * 1e3:.2f} ± {zap['uImin'] * 1e3:.2f}) mA")
        ax.axvline(zap["UZK"], color="tab:purple", ls="-.", lw=1.2,
                   label=f"$U_{{ZK}}$ = ({zap['UZK']:.3f} ± {zap['uUZK']:.3f}) V")
        ax.axvline(zap["UZ"], color="tab:brown", ls="-.", lw=1.2,
                   label=f"$U_Z$ = ({zap['UZ']:.3f} ± {zap['uUZ']:.3f}) V  (przy I = {zap['IZ'] * 1e3:.2f} mA)")
        ax.plot(zap["rs"]["U"], zap["rs"]["I"] * 1e3, "ks", ms=5, mfc="none",
                label="punkty pracy do $R_S$")
        ax.set_ylim(d["I"].min() * 1e3 * 1.08, d["I"].max() * 1e3 * 1.15)

    ax.set_xlabel("U [V]")
    ax.set_ylabel("I [mA]")
    ax.set_title(f"{NAZWA[nr]} - charakterystyka I-V (metoda punkt po punkcie)")
    ax.grid(alpha=0.4)
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(WYNIKI / f"dioda{nr}_IV.png", dpi=200)
    plt.close(fig)


def wykres_porownanie(dane):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for nr, d in dane.items():
        errorbar(ax, d, color=KOLOR[nr], ecolor=KOLOR[nr], label=NAZWA[nr])
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("U [V]")
    ax.set_ylabel("I [mA]")
    ax.set_title("Porównanie charakterystyk I-V diod (metoda punkt po punkcie)")
    ax.grid(alpha=0.4)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(WYNIKI / "porownanie_IV.png", dpi=200)
    plt.close(fig)


def zaokr(x, n=6):
    return float(f"{x:.{n}g}")


def wyniki_json(wyniki_prz, wyniki_zap):
    wynik = {
        "metoda": "punkt po punkcie",
        "zalozenia": {
            "miernik": "BRYMEN BM811 (5000 działek)",
            "dokladnosc_U": "±(0,08% + 2 cyfry)",
            "dokladnosc_I": "±(0,2% + 4 cyfry), zakres 50 mA, rozdzielczość 0,01 mA",
            "niepewnosc_standardowa": "u = dokładność / sqrt(3)",
            "prosta": "I [mA] = a * U [V] + b",
            "dopasowanie_przewodzenie_I_min_mA": I_MIN_FIT_PRZEWODZENIE,
            "dopasowanie_zaporowy_absI_min_mA": I_MIN_FIT_ZAPOROWY,
            "U_ZK": "punkt pomiarowy najbliższy przecięciu prostej (kierunek zaporowy) z osią U",
            "I_min": "prąd w punkcie U_ZK",
            "U_Z": f"napięcie w punkcie pomiarowym najbliższym |I| = {I_ZT:.0f} mA",
        },
        "diody": {},
    }
    for nr, p in wyniki_prz.items():
        dioda = {
            "nazwa": NAZWA[nr],
            "przewodzenie": {
                "a_mA_na_V": zaokr(p["a"]), "da_mA_na_V": zaokr(p["da"]),
                "b_mA": zaokr(p["b"]), "db_mA": zaokr(p["db"]),
                "R2": zaokr(p["r2"]), "n_punktow": p["n"],
                "Vbi_V": zaokr(p["Vbi"]), "u_Vbi_V": zaokr(p["uVbi"]),
            },
        }
        z = wyniki_zap[nr]
        if z:
            dioda["zaporowy"] = {
                "a_mA_na_V": zaokr(z["a"]), "da_mA_na_V": zaokr(z["da"]),
                "b_mA": zaokr(z["b"]), "db_mA": zaokr(z["db"]),
                "R2": zaokr(z["r2"]), "n_punktow": z["n"],
                "RZ_Ohm": zaokr(z["RZ"]), "u_RZ_Ohm": zaokr(z["uRZ"]),
                "przeciecie_prostej_z_osia_U_V": zaokr(z["U0"]),
                "UZK_V": zaokr(z["UZK"]), "u_UZK_V": zaokr(z["uUZK"]),
                "Imin_mA": zaokr(z["Imin"] * 1e3), "u_Imin_mA": zaokr(z["uImin"] * 1e3),
                "UZ_V": zaokr(z["UZ"]), "u_UZ_V": zaokr(z["uUZ"]),
                "IZ_mA": zaokr(z["IZ"] * 1e3), "u_IZ_mA": zaokr(z["uIZ"] * 1e3),
                "rezystancja_statyczna": [
                    {"U_V": zaokr(w["U"]), "u_U_V": zaokr(w["uU"]),
                     "I_mA": zaokr(w["I"] * 1e3), "u_I_mA": zaokr(w["uI"] * 1e3),
                     "RS_Ohm": zaokr(w["RS"]), "u_RS_Ohm": zaokr(w["uRS"])}
                    for _, w in z["rs"].iterrows()
                ],
            }
        wynik["diody"][str(nr)] = dioda
    return wynik


def wyniki_txt(wyniki_prz, wyniki_zap):
    linie = [
        "OPRACOWANIE - METODA PUNKT PO PUNKCIE",
        "=" * 78,
        "",
        "Założenia:",
        "  miernik: BRYMEN BM811 (5000 działek)",
        "  dokładność U: ±(0,08% + 2 cyfry);  dokładność I: ±(0,2% + 4 cyfry), zakres 50 mA (0,01 mA)",
        "  niepewność standardowa: u = dokładność / sqrt(3)",
        "  prosta: I [mA] = a * U [V] + b",
        f"  U_ZK - punkt pomiarowy najbliższy przecięciu prostej (|I| >= {I_MIN_FIT_ZAPOROWY:.0f} mA) z osią U",
        "  I_min - prąd w punkcie U_ZK",
        f"  U_Z  - napięcie w punkcie pomiarowym najbliższym |I| = {I_ZT:.0f} mA",
        "",
        "PODSUMOWANIE",
        "-" * 78,
        f"{'dioda':<24}{'V_bi [V]':>17}{'U_ZK [V]':>18}{'I_min [mA]':>16}{'U_Z [V]':>18}{'R_Z [Ω]':>16}",
    ]
    for nr, p in wyniki_prz.items():
        z = wyniki_zap[nr]
        vbi = f"{p['Vbi']:.3f} ± {p['uVbi']:.3f}"
        if z:
            linie.append(f"{NAZWA[nr]:<24}{vbi:>17}{z['UZK']:>10.3f} ± {z['uUZK']:.3f}"
                         f"{z['Imin'] * 1e3:>9.2f} ± {z['uImin'] * 1e3:.2f}"
                         f"{z['UZ']:>10.3f} ± {z['uUZ']:.3f}{z['RZ']:>9.2f} ± {z['uRZ']:.2f}")
        else:
            linie.append(f"{NAZWA[nr]:<24}{vbi:>17}{'brak przebicia':>18}")
    linie.append("")

    for nr, p in wyniki_prz.items():
        z = wyniki_zap[nr]
        linie += [
            "=" * 78,
            NAZWA[nr].upper(),
            "=" * 78,
            f"Kierunek przewodzenia (prosta dla I >= {I_MIN_FIT_PRZEWODZENIE:.0f} mA, n = {p['n']}):",
            f"  a = ({p['a']:.1f} ± {p['da']:.1f}) mA/V,  b = ({p['b']:.1f} ± {p['db']:.1f}) mA,"
            f"  R^2 = {p['r2']:.4f}",
            f"  V_bi = -b/a = ({p['Vbi']:.4f} ± {p['uVbi']:.4f}) V",
            "",
        ]
        if not z:
            linie += ["Kierunek zaporowy: brak przebicia w zakresie pomiaru (I ≈ 0).", ""]
            continue
        linie += [
            f"Kierunek zaporowy (prosta dla |I| >= {I_MIN_FIT_ZAPOROWY:.0f} mA, n = {z['n']}):",
            f"  a = ({z['a']:.2f} ± {z['da']:.2f}) mA/V,  b = ({z['b']:.1f} ± {z['db']:.1f}) mA,"
            f"  R^2 = {z['r2']:.4f}",
            f"  R_Z  = 1/a = ({z['RZ']:.2f} ± {z['uRZ']:.2f}) Ω",
            f"  przecięcie prostej z osią U: {z['U0']:.3f} V",
            f"  U_ZK  = ({z['UZK']:.3f} ± {z['uUZK']:.3f}) V",
            f"  I_min = ({z['Imin'] * 1e3:.2f} ± {z['uImin'] * 1e3:.2f}) mA",
            f"  U_Z   = ({z['UZ']:.3f} ± {z['uUZ']:.3f}) V  przy I_Z = ({z['IZ'] * 1e3:.2f} ± {z['uIZ'] * 1e3:.2f}) mA",
            "",
            "Rezystancja statyczna R_S = U_Z / I_Z:",
            f"  {'lp.':>4}{'U [V]':>10}{'u(U) [V]':>11}{'I [mA]':>10}{'u(I) [mA]':>11}{'R_S [Ω]':>10}{'u(R_S) [Ω]':>12}",
        ]
        for k, (_, w) in enumerate(z["rs"].iterrows(), start=1):
            linie.append(f"  {k:>4}{w['U']:>10.3f}{w['uU']:>11.4f}{w['I'] * 1e3:>10.2f}{w['uI'] * 1e3:>11.3f}"
                         f"{w['RS']:>10.2f}{w['uRS']:>12.2f}")
        linie.append("")
    return "\n".join(linie)


def main():
    WYNIKI.mkdir(exist_ok=True)
    for stary in [*WYNIKI.glob("*.png"), *WYNIKI.glob("*.csv"), *WYNIKI.glob("*.txt"), *WYNIKI.glob("*.json")]:
        stary.unlink()

    dane, wyniki_prz, wyniki_zap = {}, {}, {}
    for nr in PLIKI:
        d = wczytaj(nr)
        dane[nr] = d
        wyniki_prz[nr] = analiza_przewodzenia(d)
        wyniki_zap[nr] = analiza_zaporowa(d) if nr in DIODY_ZENERA else None
        wykres_pelny(nr, d, wyniki_prz[nr], wyniki_zap[nr])
    wykres_porownanie(dane)

    tekst = wyniki_txt(wyniki_prz, wyniki_zap)
    (WYNIKI / "wyniki.txt").write_text(tekst, encoding="utf-8")
    (WYNIKI / "wyniki.json").write_text(
        json.dumps(wyniki_json(wyniki_prz, wyniki_zap), ensure_ascii=False, indent=2), encoding="utf-8")
    print(tekst)


if __name__ == "__main__":
    main()


