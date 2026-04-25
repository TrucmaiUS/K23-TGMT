# Environment Setup

## Create a virtual environment

From the repository root:

```powershell
cd transreid_colab
python -m venv .venv
```

## Activate the virtual environment

### Windows PowerShell

```powershell
cd transreid_colab
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, allow local scripts for the current user:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Windows Command Prompt

```bat
cd transreid_colab
.venv\Scripts\activate.bat
```

### Git Bash

```bash
cd transreid_colab
source .venv/Scripts/activate
```

### Linux or macOS

```bash
cd transreid_colab
python3 -m venv .venv
source .venv/bin/activate
```

## Install dependencies

Base dependencies:

```bash
pip install -r requirements.txt
```

Optional dependencies for semantic map preparation:

```bash
pip install transformers accelerate
```

## Deactivate the virtual environment

```bash
deactivate
```

## Verify the active environment

```bash
python -c "import sys; print(sys.executable)"
```

The printed Python path should point into `.venv`.
