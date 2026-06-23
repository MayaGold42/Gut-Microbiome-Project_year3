from pathlib import Path
import nbformat

PROJECT_DIR = Path(".").resolve()
OUTPUT_FILE = PROJECT_DIR / "FULL_PROJECT_RUNNABLE.ipynb"

PY_FILES = [
    "config.py",
    "model.py",
    "model_lagged.py",
    "model_lagged_full.py",
    "stability_features_stat_models.py",
]

NOTEBOOKS = [
    "Data_Integration.ipynb",
    "Additional_Merges.ipynb",
    "Association_Testing.ipynb",
    "Association_Testing_adjusted_models.ipynb",
    "FME_Analysis_v2.ipynb",
    "FME_Analysis_v2_extended.ipynb",
    "Feature_Selection.ipynb",

    # ביקשת ששני אלה יהיו אחד אחרי השני
    "Merge_data_Analysis.ipynb",
    "Merge_python_notebook_sprint2.ipynb"
]


def make_write_file_cell(filename: str, content: str):
    code = f'''from pathlib import Path

Path({filename!r}).write_text({content!r}, encoding="utf-8")
print("Created helper file:", {filename!r})
'''
    return nbformat.v4.new_code_cell(code)


def clean_notebook_cell(cell):
    new_cell = nbformat.from_dict(cell)
    new_cell.metadata = {}
    return new_cell


def main():
    print("Working directory:")
    print(PROJECT_DIR)
    print()

    print("Files in this directory:")
    for p in sorted(PROJECT_DIR.iterdir()):
        print(" -", p.name)
    print()

    combined = nbformat.v4.new_notebook()
    combined.cells = []

    combined.cells.append(
        nbformat.v4.new_markdown_cell(
            "# Full Runnable Gut Microbiome Project\n\n"
            "This notebook combines the selected notebooks and Python files into one runnable notebook.\n\n"
            "Run this notebook from top to bottom."
        )
    )

    combined.cells.append(
        nbformat.v4.new_markdown_cell("# 1. Create helper Python files")
    )

    found_anything = False

    for filename in PY_FILES:
        path = PROJECT_DIR / filename

        if not path.exists():
            print(f"Missing Python file: {filename}")
            combined.cells.append(
                nbformat.v4.new_markdown_cell(f"## Missing Python file: `{filename}`")
            )
            continue

        found_anything = True
        print(f"Adding Python file: {filename}")

        content = path.read_text(encoding="utf-8")

        combined.cells.append(
            nbformat.v4.new_markdown_cell(f"## Python helper file: `{filename}`")
        )
        combined.cells.append(make_write_file_cell(filename, content))

    combined.cells.append(
        nbformat.v4.new_markdown_cell("# 2. Combined notebooks")
    )

    for filename in NOTEBOOKS:
        path = PROJECT_DIR / filename

        if not path.exists():
            print(f"Missing notebook: {filename}")
            combined.cells.append(
                nbformat.v4.new_markdown_cell(f"# Missing notebook: `{filename}`")
            )
            continue

        found_anything = True
        print(f"Adding notebook: {filename}")

        nb = nbformat.read(path, as_version=4)

        combined.cells.append(
            nbformat.v4.new_markdown_cell(f"# Notebook: `{filename}`")
        )

        for cell in nb.cells:
            combined.cells.append(clean_notebook_cell(cell))

    if not found_anything:
        print()
        print("ERROR: No selected files were found.")
        print("You are probably running this script from the wrong folder.")
        return

    nbformat.write(combined, OUTPUT_FILE)

    print()
    print("Created:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()