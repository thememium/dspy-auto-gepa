# Changelog

## v0.1.6 (2026-05-21)

[Compare changes](https://github.com/thememium/dspy-auto-gepa/compare/v0.1.5...v0.1.6)

### 🏡 Chore

- **pyproject.toml**: add keywords, classifiers, license, URLs and package find config ([583716a](https://github.com/thememium/dspy-auto-gepa/commit/583716a73ee1df5ad7a78654616ff0b5d8ab491f))

### Contributors

- Edward Boswell <thememium@gmail.com>

## v0.1.5 (2026-05-21)

[Compare changes](https://github.com/thememium/dspy-auto-gepa/compare/v0.1.4...v0.1.5)

### 🏡 Chore

- **pyproject.toml**: add keywords, classifiers, license, URLs and package find config ([583716a](https://github.com/thememium/dspy-auto-gepa/commit/583716a73ee1df5ad7a78654616ff0b5d8ab491f))

### Contributors

- Edward Boswell <thememium@gmail.com>

## v0.1.4 (2026-05-21)

[Compare changes](https://github.com/thememium/dspy-auto-gepa/compare/v0.1.3...v0.1.4)

### 🚀 Enhancements

- **auto-gepa**: add field inference and mapping support ([5adea50](https://github.com/thememium/dspy-auto-gepa/commit/5adea503d69c0d6d3e6d0989e8e04287d4f673bd))
- **runner**: allow custom metric generator config and out_path support ([22a0814](https://github.com/thememium/dspy-auto-gepa/commit/22a0814a96ccee7a9ac73c522b9e5cae38e66858))
- **AutoGEPA**: add build_metric method ([97f89ec](https://github.com/thememium/dspy-auto-gepa/commit/97f89ec8d9e975ccb8dfc5d969e1385eaf18b688))

### 💅 Refactors

- **data**: handle missing signature in wrapped modules ([b9f1028](https://github.com/thememium/dspy-auto-gepa/commit/b9f1028bd63b2ae6ef15125360a8a208b21831db))
- **example.py**: move LM definitions out of main ([587c03d](https://github.com/thememium/dspy-auto-gepa/commit/587c03df0b67d79164902ca198448613b40b325b))
- **dspy_auto_gepa**: allow flexible field mapping for inputs/outputs ([6a948c0](https://github.com/thememium/dspy-auto-gepa/commit/6a948c029e86fc06c788154b9f9912788337eb3d))
- **data.py**: add inference and mapping helpers for DSPy modules ([a93e288](https://github.com/thememium/dspy-auto-gepa/commit/a93e288b8199a88a063a3f63ad4d45e59a5323fc))
- **config**: make input_fields and output_fields optional and remove required validation ([93908aa](https://github.com/thememium/dspy-auto-gepa/commit/93908aabb44e0a3e859c9137f99f193690b47785))
- **datasets**: rename PreparedRun to Datasets and update API ([73958fd](https://github.com/thememium/dspy-auto-gepa/commit/73958fd788cb05c46b744fb786001dbc712bb613))

### 📖 Documentation

- add bug report issue template ([c22616c](https://github.com/thememium/dspy-auto-gepa/commit/c22616cbfa618a32f2865142b6fa4bcc3fc8570c))
- add contributing guide ([71ff8d5](https://github.com/thememium/dspy-auto-gepa/commit/71ff8d5bde30f36617cc9575c8fc6ac5da85ceb4))
- **SECURITY**: add vulnerability reporting and security best practices ([78fd61c](https://github.com/thememium/dspy-auto-gepa/commit/78fd61ca3ad12f797f380dca9fc7a6674f5b757a))
- **README**: revamp and reorganize project documentation ([08d7b98](https://github.com/thememium/dspy-auto-gepa/commit/08d7b98fbc90c4a76aaac3c075ca7bd83155ecbd))
- **readme**: add model configuration section and simplify ticket rows ([3aee51c](https://github.com/thememium/dspy-auto-gepa/commit/3aee51c10f86883005d1a2a16f36a9c527ac19bd))
- **README**: remove the explicit list fields example ([e62c95d](https://github.com/thememium/dspy-auto-gepa/commit/e62c95ddb2768c515967605dc000be383b4419a2))
- **README**: add field inference and mapping docs, update API section ([4859c02](https://github.com/thememium/dspy-auto-gepa/commit/4859c027b4b200d9bc721438d35d2563dbac2e9e))
- add dict mapping section and update examples for column mismatches ([afa7a7c](https://github.com/thememium/dspy-auto-gepa/commit/afa7a7c5a36b411389b05363dfa6d291c0a35d24))
- add advanced, basic, and medium usage guides ([340883a](https://github.com/thememium/dspy-auto-gepa/commit/340883aaeaf5b9b03bdc5fee85a4ea6e1a308f2a))

### 🏡 Chore

- add MIT license file ([98f0055](https://github.com/thememium/dspy-auto-gepa/commit/98f005579978c7f864b73267b346d9f64cb3c6c4))

### Contributors

- Edward Boswell <thememium@gmail.com>

## v0.1.3 (2026-05-21)

[Compare changes](https://github.com/thememium/dspy-auto-gepa/compare/v0.1.2...v0.1.3)

### 🚀 Enhancements

- **dspy_auto_gepa**: support dataframe‑like rows input ([54d3414](https://github.com/thememium/dspy-auto-gepa/commit/54d34144539b41db2882e57c34113e217282f46c))
- **auto_gepa**: add support for custom metric file path ([2e284dc](https://github.com/thememium/dspy-auto-gepa/commit/2e284dcaa3c8ea71a429c91b7c339ee2916abe13))
- **example.py**: add interactive force prompt when model exists and remove CLI flag ([b1ab068](https://github.com/thememium/dspy-auto-gepa/commit/b1ab0689933c20f7599442805e194ca3da733329))
- **main**: add force flag and refactor execution logic ([ff81ccb](https://github.com/thememium/dspy-auto-gepa/commit/ff81ccbd26a12e0f9390c747e45ec25cc0652574))
- **auto_gepa/runner**: add run method and task name support ([094c6d5](https://github.com/thememium/dspy-auto-gepa/commit/094c6d506b40fa39ee0859d75224db7f6aaeaf0c))

### 💅 Refactors

- **example.py**: adjust AutoGEPA API usage ([53f1706](https://github.com/thememium/dspy-auto-gepa/commit/53f1706fd4e6b24ef81e96f1c737cbd2e8b1dbc6))
- **auto_gepa/runner.py**: make rows and module optional and add task resolver ([2d1adc7](https://github.com/thememium/dspy-auto-gepa/commit/2d1adc77e1d1b64a4a6b2fc17e8faa8c98f0710c))
- **example.py**: separate metric and teacher LM definitions ([3e12c95](https://github.com/thememium/dspy-auto-gepa/commit/3e12c95c355c5698c38d123f6be20c1ada72b82a))
- simplify RunResult definition and imports in runner.py ([3f8a3a9](https://github.com/thememium/dspy-auto-gepa/commit/3f8a3a950aaed3564cbec39ddb529d7adaddf649))
- **example.py**: change result attribute access to dot notation ([b433713](https://github.com/thememium/dspy-auto-gepa/commit/b4337135447ef354b8295265584064d9d49c2d38))
- **run**: add RunResult model and update AutoGEPA.run to return it ([122fc45](https://github.com/thememium/dspy-auto-gepa/commit/122fc45d4f34d6c69458198f245f4dc429d4726e))

### 📖 Documentation

- add metric parameter to AutoGEPA.run and AutoGEPA.prepare ([82da5c7](https://github.com/thememium/dspy-auto-gepa/commit/82da5c75e430dba9254a5efbfdbb25e5f5711e98))
- **README**: clarify AutoGEPA API and examples ([22c769d](https://github.com/thememium/dspy-auto-gepa/commit/22c769dfe5de69a94c35e829b158c583c9d5bf49))
- **readme**: update example code to use attribute access for result values ([fb03691](https://github.com/thememium/dspy-auto-gepa/commit/fb03691f70732173fbcf207ba6028d050960eb4a))
- **README**: add run() usage examples and step-by-step control ([9426fb5](https://github.com/thememium/dspy-auto-gepa/commit/9426fb57982e06d39a38557d52c17dd8cd68f1ce))

### 🏡 Chore

- **deps**: add pydantic dependency for data validation ([189f9dd](https://github.com/thememium/dspy-auto-gepa/commit/189f9ddf5670d75170a4b84bb2022691fa192dd9))

### ✅ Tests

- **auto-gepa**: add tests for constructor state and simplified run ([584cb51](https://github.com/thememium/dspy-auto-gepa/commit/584cb515a661cec6689b8dbcdc0714b959857ac0))
- **test_auto_gepa**: replace MagicMock with patch.object for module.load ([759b4e2](https://github.com/thememium/dspy-auto-gepa/commit/759b4e2ddf228c80c861b48ca4a7f3f8370cfb80))
- **auto-gepa**: change test to use result attributes instead of dict keys ([56d3f22](https://github.com/thememium/dspy-auto-gepa/commit/56d3f22da85c4e765609f5dddbc8e3761ce70608))
- **test_auto_gepa.py**: add tests for model loading and force flag handling ([47453af](https://github.com/thememium/dspy-auto-gepa/commit/47453af12a2e7acdc1b8b0fa7cb11394c0507ffa))

### Other Changes

- Merge pull request #1 from thememium/eboswell/feat/simplified-runner (#1) ([0470c18](https://github.com/thememium/dspy-auto-gepa/commit/0470c1874353e34996f5f7062e2e87c32230a3b1))

### Contributors

- Edward Boswell <thememium@gmail.com>

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
