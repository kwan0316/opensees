# Frame Analysis Backend Server

A FastAPI-based backend server for two-dimensional moment frame analysis using OpenSees.

## Local Installation (Non-Global)

**IMPORTANT:** 
- All installations should be done locally using a virtual environment. Do NOT install packages globally.
- **Python Version Requirement:** You MUST use Python 3.8, 3.9, 3.10, or 3.11. Python 3.12+ is NOT compatible with openseespy on Windows.

**👉 If you only have Python 3.12+, see [QUICK_SETUP.md](QUICK_SETUP.md) for instructions to install Python 3.11.**

### Step 0: Check/Install Python 3.8-3.11

**If you have Python 3.12+ installed:**
1. Download Python 3.11 from https://www.python.org/downloads/
2. Install it (you can have multiple Python versions installed)
3. Use `py -3.11` (Windows) or `python3.11` (Linux/Mac) to specify the version

**Check your Python version:**
```bash
python --version
# Should show Python 3.8.x, 3.9.x, 3.10.x, or 3.11.x
```

### Step 1: Create Virtual Environment

**Windows PowerShell (use Python 3.8-3.11):**
```powershell
cd "opsees backend"
# If you have Python 3.11 installed, use: py -3.11 -m venv venv
# Otherwise: python -m venv venv
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows CMD (use Python 3.8-3.11):**
```cmd
cd "opsees backend"
# If you have Python 3.11 installed, use: py -3.11 -m venv venv
# Otherwise: python -m venv venv
python -m venv venv
venv\Scripts\activate.bat
```

**Linux/Mac (use Python 3.8-3.11):**
```bash
cd "opsees backend"
# Use specific version: python3.11 -m venv venv
# Or: python3 -m venv venv
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies

After activating the virtual environment, install all required packages:

```bash
pip install -r requirements.txt
```

This will install all packages locally in the virtual environment (not globally).

### Step 3: Verify Installation

```bash
pip list
python -c "import fastapi; import openseespy; print('Installation successful!')"
```

### Step 4: Configure Environment

The `.env` file is already created with default settings. You can modify it if needed:
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)

## Running the Server

Make sure you're in the virtual environment, then:

```bash
python app.py
```

Or using uvicorn directly:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The server will be available at `http://localhost:8000`

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### GET `/`
Root endpoint with API information.

### POST `/api/eigen`
Perform eigenvalue analysis.

**Request Body:**
```json
{
  "mode_tag": 6
}
```

### POST `/api/static`
Perform static analysis with specified loads.

**Request Body:**
```json
{
  "loads": [
    {"node_id": 4, "fx": 2.5, "fy": 0.0, "mz": 0.0},
    {"node_id": 7, "fx": 5.0, "fy": 0.0, "mz": 0.0}
  ],
  "n_steps": 10
}
```

### POST `/api/seismic`
Perform seismic response analysis.

**Request Body:**
```json
{
  "time": [0.0, 0.02, 0.04, ...],
  "acceleration": [0.0, 0.1, 0.2, ...],
  "damping_ratio": 0.05,
  "n_steps": 1600,
  "dt": 0.02
}
```

### POST `/api/seismic/upload`
Perform seismic analysis with uploaded ground motion file.

**Form Data:**
- `file`: Ground motion text file (two columns: time, acceleration)
- `damping_ratio`: Damping ratio (default: 0.05)
- `n_steps`: Number of analysis steps (default: 1600)
- `dt`: Time step (default: 0.02)

## Example Usage

### Using curl:

```bash
# Eigenvalue analysis
curl -X POST "http://localhost:8000/api/eigen" \
  -H "Content-Type: application/json" \
  -d '{"mode_tag": 6}'

# Static analysis
curl -X POST "http://localhost:8000/api/static" \
  -H "Content-Type: application/json" \
  -d '{
    "loads": [
      {"node_id": 4, "fx": 2.5, "fy": 0.0, "mz": 0.0},
      {"node_id": 7, "fx": 5.0, "fy": 0.0, "mz": 0.0}
    ],
    "n_steps": 10
  }'
```

## Project Structure

```
opsees backend/
├── app.py              # Main FastAPI application
├── requirements.txt    # Python dependencies
├── .env               # Environment configuration and installation instructions
├── README.md          # This file
└── venv/              # Virtual environment (created after setup)
```

## Notes

- All packages are installed locally in the virtual environment (not globally)
- The `.env` file contains both installation instructions and configuration
- The server uses CORS middleware to allow cross-origin requests
- All analysis results are returned as JSON
- For large time series data, consider using the file upload endpoint for seismic analysis

## Troubleshooting

**If you get "command not found" errors:**
- Make sure the virtual environment is activated
- Check that you're in the correct directory

**If packages fail to install:**
- Make sure you have Python 3.8+ installed
- Try upgrading pip: `pip install --upgrade pip`
- Some packages may require additional system dependencies (especially on Linux)

