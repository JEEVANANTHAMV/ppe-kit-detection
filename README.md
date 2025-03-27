# PPE Kit Detection System

A safety violation detection system that monitors for proper use of personal protective equipment (PPE) like masks, hard hats, and safety vests.

## Getting Started

```bash
streamlit run main.py
```

## Configuration Options

The system uses configurable settings that can be set in the `.env` file:

### Database Configuration

| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| DATABASE_TYPE | Database type (postgres/sqlite) | sqlite |
| DATABASE_URL | PostgreSQL connection URL | None |
| SQLITE_DB_PATH | Path to SQLite database file | violations.db |

### Detection Thresholds

| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| PERSON_CONFIDENCE_THRESHOLD | Minimum confidence threshold for person detection | 0.8 |
| EQUIPMENT_CONFIDENCE_THRESHOLD | Minimum confidence threshold for safety equipment (hard hat, mask, vest) | 0.7 |
| VIOLATION_CONFIDENCE_THRESHOLD | Minimum confidence threshold for safety violations | 0.8 |

### Adjusting Thresholds

- Increase thresholds to reduce false positives but may miss some actual violations
- Decrease thresholds to catch more potential violations but may increase false positives
- Values must be between 0.0 and 1.0

Example configuration in `.env` file:
```
# Database configuration
DATABASE_TYPE=sqlite
SQLITE_DB_PATH=./data/violations.db

# Detection thresholds
PERSON_CONFIDENCE_THRESHOLD=0.85
EQUIPMENT_CONFIDENCE_THRESHOLD=0.75
VIOLATION_CONFIDENCE_THRESHOLD=0.8
```

## Video Status Values

| Status | Description |
|--------|-------------|
| registered | Live stream registered but not yet processed |
| uploaded | Video uploaded but not yet processed |
| processing | Video currently being processed |
| done | Processing completed successfully |
| error | Error occurred during processing |
| no_config | No tenant configuration found |
| no_faces | No employee faces registered for tenant |
| tenant_inactive | Tenant is marked as inactive |
| stopped | Processing stopped manually or due to tenant becoming inactive |
