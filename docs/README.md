# Documentation for the project

This is the documentation folder for the project. It uses sphinx to generate the documentation.

## Building the documentation locally

### Step 1: Install dependencies

Navigate to the docs directory and run:

```bash
pip install -r requirements.txt
```

This will install Sphinx and other dependencies required for building the documentation.

### Step 2: Build the documentation

```bash
python -m sphinx -W --keep-going -b html . public
```

This will build the documentation in the public directory. Warnings are
treated as errors, while `--keep-going` reports all warnings in one run.

### Step 3: Host the documentation locally

```bash
python -m http.server -d public
```

This will start a local HTTP server that hosts the documentation. You can then access the documentation by navigating to http://localhost:8000 in your web browser.
