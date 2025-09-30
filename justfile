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
