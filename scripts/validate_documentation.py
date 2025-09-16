#!/usr/bin/env python3
"""
Documentation validation script for the LLM Cybersecurity Benchmark system.

This script validates all code examples, API documentation, and integration
examples to ensure they are accurate, executable, and up-to-date.
"""

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml


class DocumentationValidator:
    """Comprehensive documentation validation."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.docs_path = project_root / "docs"
        self.benchmarks_path = project_root / "benchmarks"
        self.src_path = project_root / "src"
        self.examples_path = project_root / "examples"

        self.validation_results: dict[str, Any] = {
            "files_checked": 0,
            "code_blocks_found": 0,
            "code_blocks_validated": 0,
            "errors": [],
            "warnings": [],
            "summary": {},
        }

    def validate_all(self) -> dict[str, Any]:
        """Run comprehensive validation of all documentation."""

        print("🔍 Starting comprehensive documentation validation...")
        print("=" * 60)

        # Validate documentation files
        self._validate_documentation_files()

        # Validate code examples
        self._validate_code_examples()

        # Validate API consistency
        self._validate_api_consistency()

        # Validate configuration examples
        self._validate_configuration_examples()

        # Validate integration examples
        self._validate_integration_examples()

        # Generate summary
        self._generate_validation_summary()

        return self.validation_results

    def _validate_documentation_files(self) -> None:
        """Validate that all documentation files exist and have content."""

        print("\n📄 Validating documentation files...")

        required_docs = [
            "docs/evaluation_service_api.md",
            "docs/model_service_api.md",
            "docs/integration_guide.md",
            "benchmarks/performance_results.md",
        ]

        for doc_path in required_docs:
            full_path = self.project_root / doc_path
            self.validation_results["files_checked"] = (
                int(self.validation_results["files_checked"]) + 1
            )

            if not full_path.exists():
                self.validation_results["errors"].append(f"Missing documentation file: {doc_path}")
                print(f"  ❌ {doc_path} - File not found")
                continue

            # Check file size
            file_size = full_path.stat().st_size
            if file_size < 5000:  # Less than 5KB
                self.validation_results["warnings"].append(
                    f"Documentation file seems too small: {doc_path} ({file_size} bytes)"
                )
                print(f"  ⚠️  {doc_path} - File too small ({file_size} bytes)")
            else:
                print(f"  ✅ {doc_path} - OK ({file_size} bytes)")

            # Check for basic structure
            content = full_path.read_text(encoding="utf-8")
            if not self._validate_markdown_structure(content, doc_path):
                self.validation_results["warnings"].append(
                    f"Markdown structure issues in: {doc_path}"
                )

    def _validate_markdown_structure(self, content: str, doc_path: str) -> bool:
        """Validate basic markdown structure."""

        # Check for required sections
        required_sections = {
            "evaluation_service_api.md": ["Overview", "API Reference", "Usage Examples"],
            "model_service_api.md": ["Overview", "API Reference", "Usage Examples"],
            "integration_guide.md": ["Quick Start", "Integration", "Examples"],
            "performance_results.md": ["Performance Benchmarks", "Optimization"],
        }

        doc_name = Path(doc_path).name
        if doc_name in required_sections:
            for section in required_sections[doc_name]:
                if section.lower() not in content.lower():
                    print(f"    ⚠️  Missing section: {section}")
                    return False

        # Check for code blocks
        code_blocks = re.findall(r"```(\w+)?\n(.*?)\n```", content, re.DOTALL)
        if len(code_blocks) < 3:  # Expect at least 3 code examples
            print(f"    ⚠️  Few code examples found: {len(code_blocks)}")
            return False

        return True

    def _validate_code_examples(self) -> None:
        """Extract and validate all code examples from documentation."""

        print("\n💻 Validating code examples...")

        doc_files = list(self.docs_path.glob("*.md")) + list(self.benchmarks_path.glob("*.md"))

        for doc_file in doc_files:
            print(f"\n  📝 Checking {doc_file.name}...")
            content = doc_file.read_text(encoding="utf-8")

            # Extract Python code blocks
            python_blocks = re.findall(r"```python\n(.*?)\n```", content, re.DOTALL)
            yaml_blocks = re.findall(r"```yaml\n(.*?)\n```", content, re.DOTALL)
            bash_blocks = re.findall(r"```bash\n(.*?)\n```", content, re.DOTALL)

            self.validation_results["code_blocks_found"] = int(
                self.validation_results["code_blocks_found"]
            ) + (len(python_blocks) + len(yaml_blocks) + len(bash_blocks))

            # Validate Python code blocks
            for i, code_block in enumerate(python_blocks):
                if self._validate_python_code(code_block, f"{doc_file.name}:python:{i + 1}"):
                    self.validation_results["code_blocks_validated"] = (
                        int(self.validation_results["code_blocks_validated"]) + 1
                    )

            # Validate YAML code blocks
            for i, yaml_block in enumerate(yaml_blocks):
                if self._validate_yaml_code(yaml_block, f"{doc_file.name}:yaml:{i + 1}"):
                    self.validation_results["code_blocks_validated"] = (
                        int(self.validation_results["code_blocks_validated"]) + 1
                    )

            # Validate Bash code blocks
            for i, bash_block in enumerate(bash_blocks):
                if self._validate_bash_code(bash_block, f"{doc_file.name}:bash:{i + 1}"):
                    self.validation_results["code_blocks_validated"] = (
                        int(self.validation_results["code_blocks_validated"]) + 1
                    )

    def _validate_python_code(self, code: str, location: str) -> bool:
        """Validate Python code syntax and basic structure."""

        try:
            # Parse the code to check syntax
            ast.parse(code)

            # Check for common issues
            if "import" not in code and "from" not in code and len(code.strip().split("\n")) > 3:
                self.validation_results["warnings"].append(
                    f"Python code block may be missing imports: {location}"
                )

            # Check for async/await usage
            if "async def" in code and "await" not in code:
                self.validation_results["warnings"].append(
                    f"Async function without await usage: {location}"
                )

            print(f"    ✅ Python code valid: {location}")
            return True

        except SyntaxError as e:
            self.validation_results["errors"].append(f"Python syntax error in {location}: {e}")
            print(f"    ❌ Python syntax error in {location}: {e}")
            return False

        except Exception as e:
            self.validation_results["errors"].append(
                f"Python code validation error in {location}: {e}"
            )
            print(f"    ❌ Python validation error in {location}: {e}")
            return False

    def _validate_yaml_code(self, code: str, location: str) -> bool:
        """Validate YAML code syntax."""

        try:
            yaml.safe_load(code)
            print(f"    ✅ YAML code valid: {location}")
            return True

        except yaml.YAMLError as e:
            self.validation_results["errors"].append(f"YAML syntax error in {location}: {e}")
            print(f"    ❌ YAML syntax error in {location}: {e}")
            return False

    def _validate_bash_code(self, code: str, location: str) -> bool:
        """Validate Bash code for basic issues."""

        # Check for common bash issues
        lines = code.strip().split("\n")

        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Check for potential issues
            if " = " in line and not line.startswith("export"):
                self.validation_results["warnings"].append(
                    f"Possible bash assignment issue in {location}, line {i + 1}: {line}"
                )

            # Check for unquoted variables
            if "$" in line and '"' not in line and "'" not in line:
                self.validation_results["warnings"].append(
                    f"Unquoted variable in {location}, line {i + 1}: {line}"
                )

        print(f"    ✅ Bash code valid: {location}")
        return True

    def _validate_api_consistency(self) -> None:
        """Validate API documentation against actual code."""

        print("\n🔗 Validating API consistency...")

        # Import services to check their methods
        try:
            sys.path.insert(0, str(self.src_path))

            from benchmark.interfaces.evaluation_interfaces import MetricType
            from benchmark.services.evaluation_service import EvaluationService
            from benchmark.services.model_service import ModelService

            # Check EvaluationService methods
            eval_methods = [
                m
                for m in dir(EvaluationService)
                if not m.startswith("_") and callable(getattr(EvaluationService, m))
            ]

            documented_eval_methods = [
                "initialize",
                "shutdown",
                "register_evaluator",
                "evaluate_predictions",
                "get_evaluation_history",
                "get_evaluation_summary",
                "get_available_metrics",
                "health_check",
            ]

            for method in documented_eval_methods:
                if method in eval_methods:
                    print(f"    ✅ EvaluationService.{method} - documented and exists")
                else:
                    self.validation_results["errors"].append(
                        f"Documented method missing: EvaluationService.{method}"
                    )
                    print(f"    ❌ EvaluationService.{method} - documented but missing")

            # Check ModelService methods
            model_methods = [
                m
                for m in dir(ModelService)
                if not m.startswith("_") and callable(getattr(ModelService, m))
            ]

            documented_model_methods = [
                "initialize",
                "shutdown",
                "discover_models",
                "load_model",
                "unload_model",
                "predict",
                "batch_predict",
                "get_loaded_models",
                "get_resource_usage",
                "health_check",
            ]

            for method in documented_model_methods:
                if method in model_methods:
                    print(f"    ✅ ModelService.{method} - documented and exists")
                else:
                    self.validation_results["errors"].append(
                        f"Documented method missing: ModelService.{method}"
                    )
                    print(f"    ❌ ModelService.{method} - documented but missing")

            # Check enum values
            documented_metrics = [
                "ACCURACY",
                "PRECISION",
                "RECALL",
                "F1_SCORE",
                "ROC_AUC",
                "PERFORMANCE",
                "FALSE_POSITIVE_RATE",
                "CONFUSION_MATRIX",
                "EXPLAINABILITY",
            ]

            for metric in documented_metrics:
                if hasattr(MetricType, metric):
                    print(f"    ✅ MetricType.{metric} - documented and exists")
                else:
                    self.validation_results["errors"].append(
                        f"Documented enum value missing: MetricType.{metric}"
                    )
                    print(f"    ❌ MetricType.{metric} - documented but missing")

        except ImportError as e:
            self.validation_results["errors"].append(
                f"Cannot import modules for API validation: {e}"
            )
            print(f"    ❌ Import error during API validation: {e}")

        except Exception as e:
            self.validation_results["errors"].append(f"API validation error: {e}")
            print(f"    ❌ API validation error: {e}")

    def _validate_configuration_examples(self) -> None:
        """Validate configuration examples in documentation."""

        print("\n⚙️ Validating configuration examples...")

        # Look for YAML configuration examples
        doc_files = list(self.docs_path.glob("*.md"))

        for doc_file in doc_files:
            content = doc_file.read_text(encoding="utf-8")
            yaml_blocks = re.findall(r"```yaml\n(.*?)\n```", content, re.DOTALL)

            for i, yaml_block in enumerate(yaml_blocks):
                try:
                    config = yaml.safe_load(yaml_block)

                    # Validate common configuration patterns
                    if isinstance(config, dict):
                        self._validate_config_structure(config, f"{doc_file.name}:yaml:{i + 1}")

                except Exception as e:
                    self.validation_results["errors"].append(
                        f"Configuration validation error in {doc_file.name}: {e}"
                    )

    def _validate_config_structure(self, config: dict[str, Any], location: str) -> None:
        """Validate configuration structure."""

        # Check for common configuration sections
        if "services" in config:
            services = config["services"]

            # Validate model service config
            if "model_service" in services:
                model_config = services["model_service"]

                # Check for required fields
                expected_fields = ["max_concurrent_models"]
                for field in expected_fields:
                    if field not in model_config:
                        self.validation_results["warnings"].append(
                            f"Missing expected field in model_service config: {field} ({location})"
                        )

                # Validate field types and values
                if "max_concurrent_models" in model_config:
                    value = model_config["max_concurrent_models"]
                    if not isinstance(value, int) or value <= 0:
                        self.validation_results["warnings"].append(
                            f"Invalid max_concurrent_models value: {value} ({location})"
                        )

            # Validate evaluation service config
            if "evaluation_service" in services:
                eval_config = services["evaluation_service"]

                expected_fields = ["max_concurrent_evaluations"]
                for field in expected_fields:
                    if field not in eval_config:
                        self.validation_results["warnings"].append(
                            f"Missing expected field in evaluation_service config: {field} ({location})"
                        )

        print(f"    ✅ Configuration structure valid: {location}")

    def _validate_integration_examples(self) -> None:
        """Validate integration examples."""

        print("\n🔄 Validating integration examples...")

        # Check if examples directory exists
        if self.examples_path.exists():
            example_files = list(self.examples_path.glob("*.py"))

            for example_file in example_files:
                print(f"  📝 Checking {example_file.name}...")

                try:
                    # Check syntax
                    content = example_file.read_text(encoding="utf-8")
                    ast.parse(content)

                    # Check for common patterns
                    if "async def" in content and "asyncio.run" not in content:
                        self.validation_results["warnings"].append(
                            f"Async example without asyncio.run: {example_file.name}"
                        )

                    print(f"    ✅ {example_file.name} - syntax valid")

                except SyntaxError as e:
                    self.validation_results["errors"].append(
                        f"Syntax error in {example_file.name}: {e}"
                    )
                    print(f"    ❌ {example_file.name} - syntax error: {e}")

                except Exception as e:
                    self.validation_results["errors"].append(
                        f"Error checking {example_file.name}: {e}"
                    )
                    print(f"    ❌ {example_file.name} - error: {e}")
        else:
            print("  ⚠️  Examples directory not found")

    def _generate_validation_summary(self) -> None:
        """Generate validation summary."""

        print("\n📊 Validation Summary")
        print("=" * 60)

        total_errors = len(self.validation_results["errors"])
        total_warnings = len(self.validation_results["warnings"])
        files_checked = int(self.validation_results["files_checked"])
        code_blocks_found = int(self.validation_results["code_blocks_found"])
        code_blocks_validated = int(self.validation_results["code_blocks_validated"])

        print(f"Files checked: {files_checked}")
        print(f"Code blocks found: {code_blocks_found}")
        print(f"Code blocks validated: {code_blocks_validated}")
        print(f"Errors: {total_errors}")
        print(f"Warnings: {total_warnings}")

        # Calculate success rate
        if code_blocks_found > 0:
            success_rate = (float(code_blocks_validated) / float(code_blocks_found)) * 100
            print(f"Validation success rate: {success_rate:.1f}%")

        # Display errors
        if total_errors > 0:
            print(f"\n❌ Errors ({total_errors}):")
            for error in self.validation_results["errors"]:
                print(f"  • {error}")

        # Display warnings
        if total_warnings > 0:
            print(f"\n⚠️  Warnings ({total_warnings}):")
            for warning in self.validation_results["warnings"]:
                print(f"  • {warning}")

        # Overall status
        if total_errors == 0:
            print("\n✅ Documentation validation completed successfully!")
            if total_warnings > 0:
                print(f"   ({total_warnings} warnings to review)")
        else:
            print(f"\n❌ Documentation validation failed with {total_errors} errors!")

        # Store summary
        self.validation_results["summary"] = {
            "files_checked": files_checked,
            "code_blocks_found": code_blocks_found,
            "code_blocks_validated": code_blocks_validated,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "success_rate": (float(code_blocks_validated) / max(float(code_blocks_found), 1.0))
            * 100,
            "overall_status": "success" if total_errors == 0 else "failed",
        }


def main() -> None:
    """Main entry point for documentation validation."""

    # Find project root
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    print("🚀 LLM Cybersecurity Benchmark Documentation Validator")
    print(f"📁 Project root: {project_root}")

    # Create validator and run validation
    validator = DocumentationValidator(project_root)
    results = validator.validate_all()

    # Save results
    results_file = project_root / "validation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")

    # Exit with appropriate code
    if results["summary"]["overall_status"] == "success":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
