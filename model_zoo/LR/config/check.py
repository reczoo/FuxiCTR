import polars as pl
import os

data_file = "../../../data/avazu_custom/train.csv"
# Asigură-te că path-ul este corect din directorul unde rulezi acest script
# Poți folosi os.path.abspath pentru a verifica path-ul complet
# print(os.path.abspath(data_file))

try:
    # Încearcă să citești fișierul
    # Specifică explicit delimiter-ul și dacă există header
    df = pl.read_csv(data_file, has_header=True, separator=',')

    print("Fișier citit cu succes!")
    print("Numele coloanelor:", df.columns)
    print("Primele 5 rânduri:")
    print(df.head())

    if 'C1' in df.columns:
        print("Coloana 'C1' există în DataFrame.")
    else:
        print("Coloana 'C1' NU a fost găsită în DataFrame.")

except pl.exceptions.ComputeError as e:
    print(f"Eroare la citirea fișierului cu Polars: {e}")
    print("Verifică delimitatorul, codarea sau integritatea fișierului.")
except FileNotFoundError:
    print(f"Eroare: Fișierul '{data_file}' nu a fost găsit.")
except Exception as e:
    print(f"A apărut o eroare neașteptată: {e}")