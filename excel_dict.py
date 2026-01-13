import pandas as pd
import os

def export_results_to_excel(results: dict, out_dir: str = "."):
    """
    Crea un Excel por scoring.
    Dentro, una hoja por n_bins.
    Cada hoja: columnas = window_size, best_score, y columnas de best_params.
    """
    os.makedirs(out_dir, exist_ok=True)

    for scoring, bins_dict in results.items():
        out_path = f"{out_dir}/{scoring}.xlsx"

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            for n_bins, window_dict in bins_dict.items():
                rows = []

                # window_dict: {window_size: {"best_score": ..., "best_params": {...}}}
                for window_size, payload in window_dict.items():
                    best_accuracy_test = payload.get("best_accuracy_test", None)
                    best_params = payload.get("best_params", {})

                    row = {
                        "window_size": window_size,
                        "best_accuracy_test": best_accuracy_test,
                        **best_params
                    }
                    rows.append(row)

                df = pd.DataFrame(rows)

                # Opcional: ordenar por window_size si son números
                try:
                    df = df.sort_values("window_size")
                except Exception:
                    pass

                # Nombre de hoja válido en Excel (<=31 chars, sin ciertos caracteres)
                sheet_name = str(n_bins)[:31].replace("/", "_").replace("\\", "_").replace(":", "_")

                df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"Creado: {out_path}")