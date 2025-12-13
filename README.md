# yetanothersentimentapp
I would like this to be the last sentiment analysis app on this GitHub ;]

### Sentiment analysis application
1. Before running the application, make sure to install all dependencies:
```bash
uv sync
```

2. Install gdown:
```bash
pip install gdown

or 

uv add gdown
```

3. Download the model to the root directory of the application:
```bash
gdown https://drive.google.com/uc?id=1NRZdYq5jweVRUzAZG518LMhs4E56IgxG
```

4. Unpack the model to the model directory:
```bash
unzip model.zip -d model
```

5. Run the application:
```bash
docker compose up
```

6. Test the application:
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text": "I love this product!"}'
``` 