-- SQLite Database Schema for Benchmark Results Storage
-- This schema supports storing evaluation results with efficient querying
-- and maintains data integrity across experiments, models, and datasets.

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- Schema version tracking for migrations
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL,
    description TEXT
);

-- Experiments table - groups of related evaluations
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    config_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT CHECK(status IN ('running', 'completed', 'failed', 'cancelled')) DEFAULT 'running',
    metadata TEXT  -- JSON metadata
);

-- Datasets table - information about evaluation datasets
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source TEXT NOT NULL,
    version TEXT,
    samples_count INTEGER,
    created_at TEXT NOT NULL,
    metadata TEXT,  -- JSON metadata
    file_path TEXT,
    checksum TEXT
);

-- Models table - information about evaluated models
CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    version TEXT,
    parameters_count BIGINT,
    created_at TEXT NOT NULL,
    config TEXT,  -- JSON configuration
    architecture TEXT,
    training_data TEXT,
    performance_profile TEXT  -- JSON performance characteristics
);

-- Evaluations table - individual evaluation runs
CREATE TABLE IF NOT EXISTS evaluations (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT CHECK(status IN ('running', 'completed', 'failed')) DEFAULT 'running',
    error_message TEXT,
    execution_time_seconds REAL,
    samples_processed INTEGER,
    success BOOLEAN DEFAULT 1,
    metadata TEXT  -- JSON metadata
);

-- Evaluation results table - individual metric results
CREATE TABLE IF NOT EXISTS evaluation_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_id TEXT NOT NULL REFERENCES evaluations(id) ON DELETE CASCADE,
    metric_type TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    value REAL NOT NULL,
    metadata TEXT,  -- JSON metadata for metric-specific details
    created_at TEXT NOT NULL,
    -- Ensure metric uniqueness per evaluation
    UNIQUE(evaluation_id, metric_type, metric_name)
);

-- Predictions table - individual prediction results for detailed analysis
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_id TEXT NOT NULL REFERENCES evaluations(id) ON DELETE CASCADE,
    sample_id TEXT NOT NULL,
    input_text TEXT NOT NULL,
    prediction TEXT NOT NULL,
    confidence REAL,
    explanation TEXT,
    ground_truth TEXT,
    processing_time_ms REAL,
    created_at TEXT NOT NULL,
    metadata TEXT  -- JSON for additional prediction data
);

-- Evaluation configurations table - track evaluation parameters
CREATE TABLE IF NOT EXISTS evaluation_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_id TEXT NOT NULL REFERENCES evaluations(id) ON DELETE CASCADE,
    config_key TEXT NOT NULL,
    config_value TEXT NOT NULL,
    config_type TEXT NOT NULL  -- 'string', 'number', 'boolean', 'json'
);

-- Performance indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_evaluations_experiment ON evaluations(experiment_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_model ON evaluations(model_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_dataset ON evaluations(dataset_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_status ON evaluations(status);
CREATE INDEX IF NOT EXISTS idx_evaluations_completed ON evaluations(completed_at);
CREATE INDEX IF NOT EXISTS idx_evaluations_execution_time ON evaluations(execution_time_seconds);

CREATE INDEX IF NOT EXISTS idx_evaluation_results_evaluation ON evaluation_results(evaluation_id);
CREATE INDEX IF NOT EXISTS idx_evaluation_results_metric ON evaluation_results(metric_type, metric_name);
CREATE INDEX IF NOT EXISTS idx_evaluation_results_value ON evaluation_results(value);

CREATE INDEX IF NOT EXISTS idx_predictions_evaluation ON predictions(evaluation_id);
CREATE INDEX IF NOT EXISTS idx_predictions_sample ON predictions(sample_id);
CREATE INDEX IF NOT EXISTS idx_predictions_confidence ON predictions(confidence);

CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiments_created ON experiments(created_at);

CREATE INDEX IF NOT EXISTS idx_models_type ON models(type);
CREATE INDEX IF NOT EXISTS idx_datasets_source ON datasets(source);

-- Composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_results_metric_value ON evaluation_results(metric_type, metric_name, value);
CREATE INDEX IF NOT EXISTS idx_evaluations_composite ON evaluations(experiment_id, model_id, dataset_id, status);

-- Views for common queries
CREATE VIEW IF NOT EXISTS evaluation_summary AS
SELECT
    e.id as evaluation_id,
    e.experiment_id,
    e.model_id,
    e.dataset_id,
    exp.name as experiment_name,
    m.name as model_name,
    m.type as model_type,
    d.name as dataset_name,
    e.started_at,
    e.completed_at,
    e.execution_time_seconds,
    e.samples_processed,
    e.status,
    e.success,
    COUNT(er.id) as metric_count
FROM evaluations e
JOIN experiments exp ON e.experiment_id = exp.id
JOIN models m ON e.model_id = m.id
JOIN datasets d ON e.dataset_id = d.id
LEFT JOIN evaluation_results er ON e.id = er.evaluation_id
GROUP BY e.id;

-- View for metric comparison across evaluations
CREATE VIEW IF NOT EXISTS metric_comparison AS
SELECT
    er.metric_type,
    er.metric_name,
    er.value,
    e.evaluation_id,
    e.experiment_name,
    e.model_name,
    e.model_type,
    e.dataset_name,
    e.completed_at
FROM evaluation_results er
JOIN evaluation_summary e ON er.evaluation_id = e.evaluation_id
WHERE e.status = 'completed' AND e.success = 1;

-- View for performance trends
CREATE VIEW IF NOT EXISTS performance_trends AS
SELECT
    e.model_id,
    m.name as model_name,
    e.dataset_id,
    d.name as dataset_name,
    DATE(e.completed_at) as date,
    AVG(e.execution_time_seconds) as avg_execution_time,
    AVG(CAST(e.samples_processed AS REAL) / e.execution_time_seconds) as avg_throughput,
    COUNT(*) as evaluation_count
FROM evaluations e
JOIN models m ON e.model_id = m.id
JOIN datasets d ON e.dataset_id = d.id
WHERE e.status = 'completed' AND e.success = 1
GROUP BY e.model_id, e.dataset_id, DATE(e.completed_at)
ORDER BY date;

-- Initialize schema version
INSERT OR REPLACE INTO schema_version (version, applied_at, description)
VALUES (1, datetime('now'), 'Initial schema with experiments, models, datasets, evaluations, and results');
