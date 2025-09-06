# GitHub Actions Workflows

This repository uses several GitHub Actions workflows for comprehensive CI/CD automation. All workflows are optimized for **macOS-14 (Apple Silicon)** to ensure compatibility with MLX dependencies.

## 🚀 Core Workflows

### 1. CI - Code Quality & Unit Tests (`ci.yml`)
**Triggers:** Push/PR to `main`/`develop`, manual dispatch
**Purpose:** Fast feedback loop for code quality and unit testing

**Jobs:**
- **Code Quality** - Runs on Python 3.11 & 3.12
  - Ruff linting and formatting
  - MyPy type checking
  - Bandit security scanning
- **Unit Tests** - Comprehensive unit test suite with coverage
- **Data Generators Test** - Validates cybersecurity data generation

**Key Features:**
- ⚡ Poetry dependency caching for faster builds
- 📊 Coverage reporting with Codecov integration
- 🔒 Security scanning with artifact uploads
- 🍎 Apple Silicon optimization for MLX compatibility

### 2. Integration & End-to-End Tests (`tests.yml`)
**Triggers:** Push/PR to `main`/`develop`, daily schedule (2 AM UTC), manual dispatch
**Purpose:** Comprehensive integration and system testing

**Jobs:**
- **Integration Tests** - Database, config, and component integration
- **E2E Tests** - Full system workflow simulation
- **Performance Tests** - Benchmarking data generation and DB operations

**Key Features:**
- 🔄 MLX compatibility testing on Apple Silicon
- 📈 Performance benchmarking with thresholds
- 🧪 Mock end-to-end experiment execution
- 📊 Multi-format test reporting

### 3. Security Scanning (`security.yml`)
**Triggers:** Push/PR, weekly schedule (Sunday 3 AM UTC), manual dispatch
**Purpose:** Comprehensive security vulnerability scanning

**Security Tools:**
- **Safety** - Known vulnerability database scanning
- **Bandit** - Static security analysis for Python
- **Semgrep** - Advanced static analysis (OWASP Top 10)
- **detect-secrets** - Hardcoded secret detection
- **pip-audit** - Dependency vulnerability audit

**Key Features:**
- 🔒 SARIF report generation for GitHub Security
- 📄 License compliance checking
- 🚨 Automated security alerts
- 📊 Comprehensive security reporting

### 4. Dependency Management (`dependencies.yml`)
**Triggers:** Weekly schedule (Monday 9 AM UTC), manual dispatch
**Purpose:** Automated dependency updates and security monitoring

**Capabilities:**
- 📦 Automated dependency auditing
- ⬆️ Configurable update strategies (patch/minor/major)
- 🔄 Automated PR creation for dependency updates
- 🛡️ Security vulnerability tracking
- 📋 Comprehensive dependency reporting

### 5. Release & Documentation (`release.yml`)
**Triggers:** Tag creation, release publication, manual dispatch
**Purpose:** Automated releases and documentation generation

**Features:**
- ✅ Pre-release validation (full test suite)
- 📚 Automatic API documentation generation
- 📝 Usage examples and tutorials
- 🏷️ Automated version bumping and tagging
- 📦 PyPI publishing (when configured)

## 🔧 Configuration

### Environment Variables
```yaml
POETRY_VERSION: "1.7.1"           # Poetry version for consistency
POETRY_CACHE_DIR: ~/.cache/pypoetry  # Caching directory
POETRY_VENV_IN_PROJECT: true      # Local venv for better caching
```

### Required Secrets (Optional)
```yaml
GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}  # Auto-provided
PYPI_TOKEN: ${{ secrets.PYPI_TOKEN }}      # For PyPI publishing
SEMGREP_APP_TOKEN: ${{ secrets.SEMGREP_APP_TOKEN }}  # For Semgrep Pro
```

## 📊 Workflow Status Badges

Add these badges to your main README.md:

```markdown
[![CI](https://github.com/username/LLM_Benchmark/workflows/CI%20-%20Code%20Quality%20&%20Unit%20Tests/badge.svg)](https://github.com/username/LLM_Benchmark/actions/workflows/ci.yml)
[![Tests](https://github.com/username/LLM_Benchmark/workflows/Integration%20&%20End-to-End%20Tests/badge.svg)](https://github.com/username/LLM_Benchmark/actions/workflows/tests.yml)
[![Security](https://github.com/username/LLM_Benchmark/workflows/Security%20Scanning/badge.svg)](https://github.com/username/LLM_Benchmark/actions/workflows/security.yml)
[![Dependencies](https://github.com/username/LLM_Benchmark/workflows/Dependency%20Management/badge.svg)](https://github.com/username/LLM_Benchmark/actions/workflows/dependencies.yml)
```

## 🎯 Workflow Optimization Features

### Caching Strategy
- **Poetry Installation**: Cached across runs for faster setup
- **Dependencies**: Multi-level caching with fallback restoration
- **Virtual Environments**: Cached in-project for consistency

### Performance Optimizations
- **Concurrency Control**: Prevents duplicate runs on same ref
- **Job Dependencies**: Efficient workflow orchestration
- **Selective Testing**: Targeted test execution based on triggers
- **Timeout Management**: Prevents stuck workflows

### Apple Silicon / MLX Compatibility
- **macOS-14 Runners**: Native Apple Silicon support
- **MLX Testing**: Validates MLX imports and operations
- **Performance Tuning**: Optimized for Apple Silicon ML workloads

## 🚦 Workflow Triggers Summary

| Workflow | Push | PR | Schedule | Manual | Tags |
|----------|------|----|---------|---------|----- |
| CI | ✅ | ✅ | ❌ | ✅ | ❌ |
| Tests | ✅ | ✅ | ✅ Daily | ✅ | ❌ |
| Security | ✅ | ✅ | ✅ Weekly | ✅ | ❌ |
| Dependencies | ❌ | ❌ | ✅ Weekly | ✅ | ❌ |
| Release | ❌ | ❌ | ❌ | ✅ | ✅ |

## 📝 Usage Examples

### Manual Workflow Dispatch

**Run specific test types:**
```bash
# Via GitHub CLI
gh workflow run tests.yml -f test_type=integration
gh workflow run tests.yml -f test_type=performance

# Via GitHub UI
Go to Actions → Integration & End-to-End Tests → Run workflow
```

**Trigger dependency updates:**
```bash
gh workflow run dependencies.yml -f update_type=minor
```

**Create a release:**
```bash
gh workflow run release.yml -f release_type=patch
```

## 🛠️ Local Development

To run similar checks locally:

```bash
# Code quality checks
poetry run ruff check src/ tests/
poetry run mypy src/ tests/

# Unit tests with coverage
poetry run pytest tests/unit/ --cov=src/benchmark

# Integration tests
poetry run pytest tests/integration/

# Security scanning
poetry run bandit -r src/
poetry run safety check
```

## 🔍 Troubleshooting

### Common Issues

1. **MLX Import Failures**: Expected in CI - MLX may not be available in GitHub Actions
2. **Cache Misses**: Check Poetry version consistency across workflows
3. **Test Timeouts**: Adjust timeout values in workflow files
4. **Dependency Conflicts**: Use `poetry lock --no-update` to resolve

### Debug Mode

Add this step to any workflow for debugging:
```yaml
- name: Debug Environment
  run: |
    echo "Python version: $(python --version)"
    echo "Poetry version: $(poetry --version)"
    echo "Installed packages:"
    poetry show
    echo "Environment variables:"
    env | sort
```

## 📈 Monitoring & Metrics

The workflows generate comprehensive artifacts and reports:
- Test results (JUnit XML)
- Coverage reports (XML, HTML)
- Security scan results (JSON, SARIF)
- Dependency reports (JSON, CSV)
- Performance benchmarks
- Build artifacts

All artifacts are retained for 30 days and uploaded to GitHub Actions for review.
