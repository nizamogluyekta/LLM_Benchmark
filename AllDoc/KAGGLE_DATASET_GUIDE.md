# Using Kaggle Cybersecurity Datasets with LM Studio Benchmark

This guide shows you how to use real Kaggle cybersecurity datasets with your LM Studio few-shot learning benchmark.

## Quick Start

### Option 1: Download and Use Kaggle Dataset

```bash
# Example with a CSV dataset
python3 lm_studio_kaggle_benchmark.py \
    --model-name "meta-llama-3.1-8b-instruct" \
    --kaggle-dataset "path/to/your/dataset.csv" \
    --text-column "description" \
    --label-column "is_malicious"
```

### Option 2: Use Built-in Synthetic Data

```bash
# Use synthetic cybersecurity data (no download needed)
python3 lm_studio_kaggle_benchmark.py \
    --model-name "meta-llama-3.1-8b-instruct" \
    --use-synthetic \
    --max-samples 200
```

## Recommended Kaggle Datasets

Here are some excellent cybersecurity datasets you can download from Kaggle:

### 1. **Network Intrusion Detection**
- **Dataset**: [NSL-KDD Dataset](https://www.kaggle.com/datasets/hassan06/nslkdd)
- **Description**: Network intrusion detection with various attack types
- **Columns**: Features + `class` (normal/attack_type)
- **Usage**:
  ```bash
  python3 lm_studio_kaggle_benchmark.py \
      --model-name "your-model" \
      --kaggle-dataset "KDDTrain+.txt" \
      --text-column "protocol_type,service,flag" \
      --label-column "class"
  ```

### 2. **Malware Detection**
- **Dataset**: [Malware Detection](https://www.kaggle.com/datasets/xwolf12/malware-detection)
- **Description**: Static analysis features for malware classification
- **Columns**: Features + `Class` (0=benign, 1=malware)
- **Usage**:
  ```bash
  python3 lm_studio_kaggle_benchmark.py \
      --model-name "your-model" \
      --kaggle-dataset "malware_data.csv" \
      --text-column "Name" \
      --label-column "Class"
  ```

### 3. **Phishing Websites**
- **Dataset**: [Phishing Websites](https://www.kaggle.com/datasets/akashkr/phishing-website-dataset)
- **Description**: Features of phishing vs legitimate websites
- **Columns**: Multiple features + `Result` (-1=phishing, 1=legitimate)
- **Usage**:
  ```bash
  python3 lm_studio_kaggle_benchmark.py \
      --model-name "your-model" \
      --kaggle-dataset "phishing.csv" \
      --text-column "URL" \
      --label-column "Result"
  ```

### 4. **CICIDS2017**
- **Dataset**: [CICIDS2017](https://www.kaggle.com/datasets/cicdataset/cicids2017)
- **Description**: Comprehensive intrusion detection dataset
- **Columns**: Network flow features + `Label`
- **Usage**:
  ```bash
  python3 lm_studio_kaggle_benchmark.py \
      --model-name "your-model" \
      --kaggle-dataset "cicids2017.csv" \
      --text-column "Flow_ID" \
      --label-column "Label"
  ```

### 5. **UNSW-NB15**
- **Dataset**: [UNSW-NB15](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15)
- **Description**: Network intrusion dataset with modern attack types
- **Columns**: Network features + `label` (0=normal, 1=attack)
- **Usage**:
  ```bash
  python3 lm_studio_kaggle_benchmark.py \
      --model-name "your-model" \
      --kaggle-dataset "UNSW_NB15.csv" \
      --text-column "service" \
      --label-column "label"
  ```

## Dataset Preparation

### Step 1: Download from Kaggle

1. Go to [Kaggle](https://www.kaggle.com)
2. Search for cybersecurity datasets
3. Download the CSV/JSON file
4. Place it in your benchmark directory

### Step 2: Identify Column Names

Look at your dataset to identify:
- **Text column**: Contains the data to analyze (descriptions, URLs, etc.)
- **Label column**: Contains the classification (attack/normal, malicious/benign, etc.)

```bash
# Quick preview of your dataset
head -5 your_dataset.csv
```

### Step 3: Run the Benchmark

```bash
python3 lm_studio_kaggle_benchmark.py \
    --model-name "your-lm-studio-model" \
    --kaggle-dataset "your_dataset.csv" \
    --text-column "description_column" \
    --label-column "target_column" \
    --max-samples 500 \
    --few-shot-examples 15
```

## Supported Label Formats

The script automatically normalizes various label formats to ATTACK/BENIGN:

### Attack Labels (→ ATTACK)
- `attack`, `malicious`, `malware`, `phishing`
- `intrusion`, `anomaly`, `suspicious`, `threat`
- `dos`, `ddos`, `injection`, `backdoor`
- `1`, `true`, `yes`, `positive`

### Benign Labels (→ BENIGN)
- `normal`, `benign`, `legitimate`, `clean`
- `safe`, `regular`, `baseline`, `ok`
- `0`, `false`, `no`, `negative`

## Dataset Format Examples

### CSV Format
```csv
text,label,additional_info
"SQL injection attempt detected in login form",attack,high_severity
"User successfully logged in from office network",normal,low_risk
"Multiple failed login attempts from suspicious IP",malicious,medium_severity
```

### JSON Format
```json
[
  {
    "description": "Unusual network traffic pattern detected",
    "classification": "anomaly",
    "severity": "high"
  },
  {
    "description": "Regular backup operation completed",
    "classification": "normal",
    "severity": "low"
  }
]
```

## Advanced Usage

### Custom Dataset Processing

```bash
# Limit dataset size and customize few-shot examples
python3 lm_studio_kaggle_benchmark.py \
    --model-name "llama-3.1-8b" \
    --kaggle-dataset "large_dataset.csv" \
    --text-column "event_description" \
    --label-column "threat_level" \
    --max-samples 1000 \
    --few-shot-examples 20
```

### Multiple Column Text

If your dataset has multiple text columns, you can concatenate them:

```python
# Preprocessing step (run separately)
import pandas as pd
df = pd.read_csv('dataset.csv')
df['combined_text'] = df['protocol'] + ' ' + df['service'] + ' ' + df['flag']
df.to_csv('processed_dataset.csv', index=False)
```

Then use:
```bash
python3 lm_studio_kaggle_benchmark.py \
    --kaggle-dataset "processed_dataset.csv" \
    --text-column "combined_text" \
    --label-column "class"
```

## Output Reports

The benchmark generates two reports:

1. **Detailed JSON Report** (`kaggle_benchmark_YYYYMMDD_HHMMSS.json`)
   - Complete predictions and metrics
   - Training examples used
   - Processing timestamps

2. **Human-Readable Summary** (`kaggle_summary_YYYYMMDD_HHMMSS.txt`)
   - Performance metrics
   - Dataset information
   - Sample predictions

## Troubleshooting

### Common Issues

1. **Column Not Found**
   ```
   Error: Text column 'description' not found
   ```
   **Solution**: Check column names with `head -1 your_dataset.csv`

2. **Label Normalization Failed**
   ```
   Warning: Could not normalize label: custom_label
   ```
   **Solution**: The script will skip unrecognized labels. Preprocess to standard format.

3. **Memory Issues with Large Datasets**
   ```
   Error: Memory error loading dataset
   ```
   **Solution**: Use `--max-samples` to limit dataset size

### Dataset Validation

```bash
# Check your dataset structure first
python3 -c "
import pandas as pd
df = pd.read_csv('your_dataset.csv')
print('Columns:', list(df.columns))
print('Shape:', df.shape)
print('Label distribution:')
print(df['your_label_column'].value_counts())
"
```

## Tips for Best Results

1. **Balanced Datasets**: The script automatically balances classes for fair evaluation
2. **Text Quality**: Datasets with descriptive text work better than just feature vectors
3. **Sample Size**: 200-1000 samples usually provide good evaluation results
4. **Few-Shot Examples**: 10-20 examples per prediction work well

## Example Complete Workflow

```bash
# 1. Download dataset from Kaggle (e.g., malware detection)
# 2. Check the structure
head -5 malware_dataset.csv

# 3. Run benchmark
python3 lm_studio_kaggle_benchmark.py \
    --model-name "meta-llama-3.1-8b-instruct" \
    --kaggle-dataset "malware_dataset.csv" \
    --text-column "filename" \
    --label-column "is_malware" \
    --max-samples 400 \
    --few-shot-examples 12

# 4. Review results in generated reports
```

This approach gives you real-world cybersecurity evaluation using actual attack data from Kaggle, providing more realistic assessment of your LM Studio model's cybersecurity detection capabilities!
