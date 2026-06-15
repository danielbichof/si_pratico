import shutil
import subprocess
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


BASE_DIR = Path(__file__).resolve().parent

HEART_URL = "https://archive.ics.uci.edu/static/public/519/heart%2Bfailure%2Bclinical%2Brecords.zip"
WINE_URL = "https://archive.ics.uci.edu/static/public/186/wine%2Bquality.zip"

HEART_DIR = BASE_DIR / "questao1_heart_failure"
WINE_DIR = BASE_DIR / "questao2_wine_quality"
BLACK_FRIDAY_DIR = BASE_DIR / "questao3_black_friday"


def baixar_zip(url, destino_zip):
    destino_zip.parent.mkdir(parents=True, exist_ok=True)
    print("Baixando:", url)
    urlretrieve(url, destino_zip)
    print("Arquivo salvo em:", destino_zip)


def extrair_zip(arquivo_zip, destino):
    with zipfile.ZipFile(arquivo_zip) as pacote:
        pacote.extractall(destino)
    print("Arquivos extraidos em:", destino)


def baixar_heart_failure():
    arquivo_zip = HEART_DIR / "heart_failure.zip"
    csv_destino = HEART_DIR / "heart_failure_clinical_records_dataset.csv"

    if csv_destino.exists():
        print("Heart Failure ja existe em:", csv_destino)
        return

    baixar_zip(HEART_URL, arquivo_zip)
    extrair_zip(arquivo_zip, HEART_DIR)


def baixar_wine_quality():
    arquivo_zip = WINE_DIR / "wine_quality.zip"
    red = WINE_DIR / "winequality-red.csv"
    white = WINE_DIR / "winequality-white.csv"

    if red.exists() and white.exists():
        print("Wine Quality ja existe em:", WINE_DIR)
        return

    baixar_zip(WINE_URL, arquivo_zip)
    extrair_zip(arquivo_zip, WINE_DIR)


def baixar_black_friday():
    destino_csv = BLACK_FRIDAY_DIR / "black_friday_sales.csv"
    if destino_csv.exists():
        print("Black Friday ja existe em:", destino_csv)
        return

    if not shutil.which("kaggle"):
        print("Kaggle CLI nao encontrada.")
        print("Instale com `pip install kaggle` ou coloque o CSV manualmente em:")
        print(destino_csv)
        return

    comando = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        "noopurbhatt/retail-black-friday-sales-dataset",
        "-p",
        str(BLACK_FRIDAY_DIR),
        "--unzip",
    ]

    try:
        subprocess.run(comando, check=True)
    except subprocess.CalledProcessError:
        print("Nao foi possivel baixar o dataset Black Friday pelo Kaggle.")
        print("Configure credenciais em `~/.kaggle/kaggle.json` ou coloque o CSV em:")
        print(destino_csv)
        return

    csvs = sorted(BLACK_FRIDAY_DIR.glob("*.csv"), key=lambda path: path.stat().st_size, reverse=True)
    if csvs and not destino_csv.exists():
        csvs[0].rename(destino_csv)
        print("CSV renomeado para:", destino_csv)


if __name__ == "__main__":
    baixar_heart_failure()
    baixar_wine_quality()
    baixar_black_friday()
