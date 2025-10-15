# list recipes
default:
    just --list

# run main program
run:
    uv run main.py

# install dependencies
install:
    uv sync --frozen

# update dependencies
update:
    uv lock --upgrade


# check code
check:
    ruff check

# format project
format:
    ruff format

# lager rapporter ved å konvertere .py til .ipynb filer
@convert_reports:
    echo "Converting python files to ipython notebooks"; \
    for f in $(find src/brev_analyse/ -type f -name "*.py"); do \
        p2j -o "$f"; \
    done