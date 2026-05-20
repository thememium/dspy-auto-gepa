# Changelog

## v0.1.1 (2026-05-20)

### 🚀 Enhancements

- **dspy_auto_gepa**: add per‑task artifact directories and simplify metric handling ([2397ea2](https://github.com/thememium/dspy-auto-gepa/commit/2397ea2f30fed1430067e61e7186f410675e3cfb))
- **example**: add example usage script for ticket classification ([af83e6d](https://github.com/thememium/dspy-auto-gepa/commit/af83e6dec13e258e860f6ef90b4ae335fac0cf79))
- add optional metric_name and force parameters to AutoGEPA.prepare ([9655267](https://github.com/thememium/dspy-auto-gepa/commit/9655267e72206c75f141a15d00f1b75b9354c831))
- **dspy_auto_gepa**: add AutoGEPA runner for scaffold, train, compare ([efedb0f](https://github.com/thememium/dspy-auto-gepa/commit/efedb0f4da6d9ce77b6f8a80d5d3ffdf3ddb0c58))
- **dspy_auto_gepa**: add metric builder for GEPA optimization ([fedb3e5](https://github.com/thememium/dspy-auto-gepa/commit/fedb3e5b4550fdd9508d4d0b3bade549f39554c3))
- **data**: add DSPY example conversion and split helpers ([1265124](https://github.com/thememium/dspy-auto-gepa/commit/1265124fee23fbe6fd4dc0938441fbf4fffb6224))
- **dspy_auto_gepa**: add AutoGEPAConfig dataclass ([7a6f9fb](https://github.com/thememium/dspy-auto-gepa/commit/7a6f9fb32887f9540387b420161d304c8acfb81a))
- **dspy_auto_gepa**: add artifacts module for dynamic metric loading and result persistence ([e43324e](https://github.com/thememium/dspy-auto-gepa/commit/e43324e60671c5a63fb07db3f9692b11b675b2c6))
- **pyproject.toml**: add DSPy dependency and description to package metadata ([9de6b16](https://github.com/thememium/dspy-auto-gepa/commit/9de6b1680ffae30ed09ce0db11efd49b140d785d))

### 💅 Refactors

- **config.py**: update metric and reflection default models to new openrouter providers ([382d832](https://github.com/thememium/dspy-auto-gepa/commit/382d832d74fca8bdfd8af9588c8fff93d136dbbb))
- **AutoGEPA**: rename scaffold to prepare and metric_path to metric_file ([f265b73](https://github.com/thememium/dspy-auto-gepa/commit/f265b73101463538833ef14e6f9d62143a1b74a4))
- **dspy_auto_gepa/__init__.py**: expose public API ([adcf38f](https://github.com/thememium/dspy-auto-gepa/commit/adcf38f602f37453cd842d90a4adf770ec94df83))

### 📖 Documentation

- update name param description and example usage ([6576ae7](https://github.com/thememium/dspy-auto-gepa/commit/6576ae7da76d756bfdf97fd08cb541155852677d))
- **readme**: add detailed README with installation, usage, and API ([0d6f83f](https://github.com/thememium/dspy-auto-gepa/commit/0d6f83fc0b972612e693530882e9cc0a6e0a61cd))

### 🏡 Chore

- add .auto_gepa to .gitignore ([8b055c2](https://github.com/thememium/dspy-auto-gepa/commit/8b055c29134bef77e972784a551b1eb8563a147e))
- **deps**: update dspy to 3.2.1 ([7ccd66f](https://github.com/thememium/dspy-auto-gepa/commit/7ccd66fc280845391bcc2ccab90f16e4950050aa))

### ✅ Tests

- add tests for AutoGEPA config and helper functions ([ada7a01](https://github.com/thememium/dspy-auto-gepa/commit/ada7a015065f3834bf8a8aaa8c7dde7df1852910))

### Contributors

- Edward Boswell <thememium@gmail.com>
