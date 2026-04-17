# ml_ci_pipeline_demo

A minimal ML project demonstrating a production-style CI/CD pipeline with containerization. Built as a hands-on learning project to illustrate software engineering best practices for machine learning codebases.

---

## What this project demonstrates

- Clean ML repo structure with model definition, training script, and tests separated into distinct modules
- Automated CI pipeline using GitHub Actions with fast-fail job ordering
- Docker containerization with layer caching and lean production images
- pytest-based model validation via forward pass testing
- Ruff linting enforced on every push and pull request

---

## Project structure

```
ml_ci_pipeline_demo/
├── .github/
│   └── workflows/
│       └── workflow.yml        # CI pipeline definition
├── tests/
│   └── test_forward_pass.py    # Forward pass smoke test
├── model.py                    # IrisClassificationNetwork (MLP, PyTorch)
├── training.py                 # Full training loop with eval and model saving
├── Dockerfile                  # Container definition
├── .dockerignore
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Dev/test dependencies
└── conftest.py                 # pytest path configuration
```

---

## Model

A simple MLP classifier trained on the Iris dataset:

- Input: 4 features (sepal/petal length and width)
- Architecture: Linear(4→32) → ReLU → Linear(32→32) → ReLU → Linear(32→3)
- Output: 3 classes (Iris species)
- Loss: CrossEntropyLoss
- Optimizer: Adam

---

## CI pipeline

The GitHub Actions workflow runs on every push and pull request to `main`:

```
lint-job  ──►  test-forward-pass-job
```

- `lint-job` builds the Docker image and runs `ruff check .` inside the container
- `test-forward-pass-job` runs only if lint passes, mounts the test suite, and executes `pytest tests/`
- Both jobs run entirely inside Docker — no dependency installation on the runner

---

## Running locally

**Build the image:**
```bash
docker build -t ml-ci-demo .
```

**Run training:**
```bash
docker run ml-ci-demo python training.py
```

**Run tests:**
```bash
docker run -v $(pwd)/tests:/app/tests -v $(pwd)/conftest.py:/app/conftest.py ml-ci-demo pytest tests/
```

**Run linter:**
```bash
docker run ml-ci-demo ruff check .
```

---

## Dependencies

Production (`requirements.txt`): `torch` (CPU only), `scikit-learn`, `tqdm`

Dev (`requirements-dev.txt`): `pytest`, `ruff`
