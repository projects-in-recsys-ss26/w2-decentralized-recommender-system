# Backend Decentralized Recommender Systems

## How to install the dependencies

In this project, we use uv, so install uv or choose your favorite llm und transfer this readme to a different package manager.

```
# clone the project
git clone https://github.com/mattes-kraus/masterproject-decentralized-recommender-systems.git

# navigate to the root of the project
cd masterproject-decentralized-recommender-systems/backend

# install dependencies
uv venv
uv sync
```

## How to run the recommending system backend

``` 
# navigate to masterproject/backend

# activate the venv
# windows:
.venv/Scripts/activate

# linux/ mac
source .venv/bin/activate

# run the backend
python api.py
```