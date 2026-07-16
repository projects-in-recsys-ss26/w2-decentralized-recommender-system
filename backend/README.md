# Backend - Decentralized Recommender Systems

This is the backend component for the Decentralized Recommender Systems prototype. It serves as the core API, providing recommendations based on the included models (`category_gossip` and `FedKG`).

## Tech Stack
This project is built using:
- **Python 3.13+**
- **FastAPI** (Web framework for building APIs)
- **PyTorch** (Machine Learning framework used for models like FedKG)
- **Pandas & Scikit-learn** (Data processing and analysis)
- **uv** (Extremely fast Python package installer and resolver)

## Prerequisites
Before you begin, ensure you have the following installed on your machine:
- [Python 3.13 or newer](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/) (Recommended for dependency management)

## Getting Started

Follow these steps to set up and run the backend locally:

1. **Clone the project (if you haven't already):**
   ```bash
   git clone https://github.com/mattes-kraus/masterproject-decentralized-recommender-systems.git
   ```

2. **Navigate to the backend directory:**
   ```bash
   cd masterproject-decentralized-recommender-systems/backend
   ```

3. **Install dependencies using `uv`:**
   First, create a virtual environment, and then sync the dependencies.
   ```bash
   uv venv
   uv sync
   ```
   *(Note: If you prefer a different package manager like `pip`, you can create a virtual environment and install dependencies manually based on the `pyproject.toml` file.)*

## Running the API Server

1. **Activate the virtual environment:**
   - **Windows:**
     ```bash
     .venv\Scripts\activate
     ```
   - **Linux / macOS:**
     ```bash
     source .venv/bin/activate
     ```

2. **Run the backend server:**
   Start the FastAPI server by running the main Python script:
   ```bash
   python api.py
   ```

## API Documentation

Once the server is running, FastAPI automatically generates interactive API documentation. 
You can view and test the API endpoints by navigating to:
- **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

## Retraining the Models

If you want to train the recommendation models from scratch or update them with new data, you can do so by running the respective training scripts. Ensure your virtual environment is activated before running these commands.

### 1. Retraining the Category Gossip Model
The category gossip model uses a decentralized approach (Gossip + LDP) along with user clustering. To retrain it:

```bash
# Navigate to the category_gossip directory
cd category_gossip

# Run the training script
python main.py
```
This script will process the data, train the model, evaluate it, and save the updated `.pkl` model files (e.g., `trained_model.pkl` and `user_clustering_model.pkl`) in the `category_gossip` directory.

### 2. Retraining the FedKG Model
FedKG is a federated learning framework for POI recommendation. To retrain this model:

```bash
# Navigate to the FedKG directory
cd FedKG

# Run the training script with the tuned configuration
python main.py --config_path configs_tuned.json
```
This will start the federated training process (including server-client knowledge distillation). The best models and logs will be saved in the `FedKG/output` and `FedKG/logs` directories. For more detailed instructions on FedKG (like running with/without GANs), please refer to the `backend/FedKG/README.md`.
