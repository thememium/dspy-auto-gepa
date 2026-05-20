# Changelog

## v0.1.2 (2026-05-20)

[Compare changes](https://github.com/thememium/dspy-auto-gepa/compare/v0.1.1...v0.1.2)

### 🚀 Enhancements

- **example**: add new ticket examples ([927b1b8](https://github.com/thememium/dspy-auto-gepa/commit/927b1b85410e433525610c6bcfbbf23add9e1075))
- **example**: split LM into metric_lm and reflection_lm; enable separate config ([633a405](https://github.com/thememium/dspy-auto-gepa/commit/633a405fcdb786060c51b2251e739d4778516a52))
- **metric_builder**: allow passing custom LM for metric generation ([8c92506](https://github.com/thememium/dspy-auto-gepa/commit/8c9250695ccd78748d0d24c0d6f2891e0df1e0a8))
- **example.py**: add provider order & fallback config to LM ([735ed85](https://github.com/thememium/dspy-auto-gepa/commit/735ed85536959679eb53ee9ebd03dbf703673b44))
- **dspy_auto_gepa**: add PreparedRun helper and refactor run methods ([672b6b0](https://github.com/thememium/dspy-auto-gepa/commit/672b6b034776d002392a2817148edd7d392510ca))

### 💅 Refactors

- **metric_builder.py**: replace ChainOfThought with RLM ([af371e9](https://github.com/thememium/dspy-auto-gepa/commit/af371e925c774409dcad51592ee31e4d135b74e0))
- **config**: remove duplicate LM default assignments ([bb42883](https://github.com/thememium/dspy-auto-gepa/commit/bb42883f4586619c8c05bd80ef1621fe642a6b4f))
- **AutoGEPA**: change constructor to use keyword arguments and ([4bafb37](https://github.com/thememium/dspy-auto-gepa/commit/4bafb37dcfc29cc899139869ca28518ff17dd086))
- **config**: move LM defaults to __post_init__ for optional injection ([252ab8c](https://github.com/thememium/dspy-auto-gepa/commit/252ab8c12ec9c49d7efc6beb2120b1f4ed1f194b))
- **example.py**: simplify imports & warnings, streamline AutoGEPA ([be576ba](https://github.com/thememium/dspy-auto-gepa/commit/be576ba5769a16d51f87dfd274ae977115e9ae0f))
- **example.py**: rename metric_lm var and disable cache on teacher LM ([8bf1ef5](https://github.com/thememium/dspy-auto-gepa/commit/8bf1ef59b76d3e26412bfe3a010d405e31257c7a))
- **metric_builder**: simplify generator invocation and add syntax validation ([f93db7f](https://github.com/thememium/dspy-auto-gepa/commit/f93db7f479cbdb09f6a96573ee48d799b6dfa2d7))
- **runner.py**: remove redundant dspy.context wrapper around generate_metric_file ([e24be43](https://github.com/thememium/dspy-auto-gepa/commit/e24be4304bb359fbe3f890f7224e2471450f65f8))
- **config**: replace model name strings with dspy.LM instances ([d160d0f](https://github.com/thememium/dspy-auto-gepa/commit/d160d0f27b973f54c61e6d1ab2bd7c103ed706da))
- **dspy_auto_gepa/runner.py**: use LM context for metric generation ([7935b28](https://github.com/thememium/dspy-auto-gepa/commit/7935b28e63224a5029d734fbdf660e5dcfd10935))
- **example**: switch LM to GPT OSS 120B ([5fde16d](https://github.com/thememium/dspy-auto-gepa/commit/5fde16d77832bdc722ef124ade9d6c9df1ac4d13))
- **example.py**: simplify metric handling and use prepared config ([5e639a6](https://github.com/thememium/dspy-auto-gepa/commit/5e639a652c13c56f034d2176e05ed4a4c80914cd))

### 📖 Documentation

- **readme**: rename model variables and unify large_lm usage ([ab7dde8](https://github.com/thememium/dspy-auto-gepa/commit/ab7dde85f84286c28e5b29b28325ac76e6067907))
- **readme**: update AutoGEPA constructor docs and examples ([370f7dc](https://github.com/thememium/dspy-auto-gepa/commit/370f7dc1563e1be020a2a9098d5fb4cdc035b864))
- update AutoGEPA usage examples to use PreparedRun ([fbfb649](https://github.com/thememium/dspy-auto-gepa/commit/fbfb6495c9a1f1696450ab110ec3e7c94b80dd8b))

### 🏡 Chore

- **deps**: add pandas to dependencies and silence deptry DEP002 ([fc7613f](https://github.com/thememium/dspy-auto-gepa/commit/fc7613f6fffccfa8cef5f81aaf82053a9e8e398d))

### ✅ Tests

- add test for AutoGEPA constructor to verify config values ([c28cb63](https://github.com/thememium/dspy-auto-gepa/commit/c28cb637bd1a92657aea8f64b427c9f04cc53375))

### 🎨 Styles

- **config.py**: drop trailing blank lines ([dcb3f2b](https://github.com/thememium/dspy-auto-gepa/commit/dcb3f2bbce092fcfbe94ecf1ca4571478c2eef9b))

### Contributors

- Edward Boswell <thememium@gmail.com>

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
