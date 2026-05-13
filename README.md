# Practica 2 - Repo 2: API

API REST local para servir el modelo final de la Practica 2.

La API carga tres artefactos ya fitteados y los encadena en cada prediccion:

```text
df_raw -> preprocessor.transform -> filter.transform -> practica2_model.predict_proba
```

Tambien obtiene el intervalo de incertidumbre del modelo final y deriva a un agente cuando:

```text
p_high - p_low > 0.2
```

## Estructura

```text
repo2_api_practica2/
├── api/
│   └── main.py
├── artifacts/
│   ├── preprocessor.pkl
│   ├── filter.pkl
│   ├── practica2_model.pkl
│   └── feature_schema.json
├── examples/
│   └── sample_payload.json
├── src/
│   ├── preprocessing/
│   ├── filtering/
│   └── practica2_model.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## Instalacion

Desde la raiz del repo:

```bash
uv sync
```

## Ejecucion local

```bash
uv run uvicorn api.main:app --reload --port 8080
```

La documentacion Swagger queda disponible en:

```text
http://localhost:8080/docs
```

## Endpoints

### Health check

```bash
curl http://localhost:8080/health
```

### GET /health

Verifica que la API está funcionando y devuelve la versión del modelo.

```bash
curl http://localhost:8080/health

#Respuesta esperada:

{
  "status": "ok",
  "model_version": "initial",
  "loaded_at": "2026-01-15T10:30:00Z"
}
```

### POST /predict

Recibe un JSON con las features crudas del cliente.

```bash
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  --data @examples/sample_payload.json
```

Respuesta esperada:

```json
{
  "p_default": 0.31,
  "p_low": 0.22,
  "p_high": 0.41,
  "decision": "auto",
  "reason": "p_high - p_low <= 0.2"
}
```

### POST /model/upload

Subir un nuevo modelo como multipart/form-data:

```bash
curl -X POST "http://localhost:8080/model/upload" \
  -F "file=@artifacts/practica2_model.pkl"
```

Tambien se puede cargar un modelo desde una ruta local:

```bash
curl -X POST "http://localhost:8080/model/upload" \
  -F "model_path=$(pwd)/artifacts/practica2_model.pkl"
```

La API valida que el modelo tenga los metodos `predict`, `predict_proba` y `predict_interval` antes de dejarlo activo en memoria. El wrapper del modelo soporta intervalos conformales globales y por bins de probabilidad, segun el artefacto generado en el Repo 1.

## Notas

- No se re-fittea ningun artefacto en la API.
- No se loguea el payload completo del cliente.
- Los `.pkl` se cargan localmente; no hace falta desplegar en nube.
