# Changelog

## v0.1.9 (2026-07-04)

[Compare changes](https://github.com/thememium/dspy-auto-gepa/compare/v0.1.8...v0.1.9)

### 🚀 Enhancements

- **example**: add combined AutoData + AutoGEPA pipeline script ([bb83f75](https://github.com/thememium/dspy-auto-gepa/commit/bb83f75b2a1e1b8e0641809d995032f584a4c772))
- **poe**: add example‑full task ([2ea2fcf](https://github.com/thememium/dspy-auto-gepa/commit/2ea2fcf3c352f16ef43e6ad9a7ece2d5c9b409b9))
- **poe**: add example data split and signature tasks ([b182fdb](https://github.com/thememium/dspy-auto-gepa/commit/b182fdb8fda54ca258838e3aa9cfa078df8eb037))
- **generator**: add diversity_categories field and simplify generate API ([612821d](https://github.com/thememium/dspy-auto-gepa/commit/612821da7c4e78fc7b1d6aa90282d2c2ef7ef8ac))
- **config**: add diversity, seed examples, output path and force flag ([8bf14f8](https://github.com/thememium/dspy-auto-gepa/commit/8bf14f84713465b5553a73cdbe089457ea420cad))
- **generator**: add signature generation mode ([ba4e078](https://github.com/thememium/dspy-auto-gepa/commit/ba4e078b8336c21650a6d576c1d5be39440bf653))
- **config**: add generation_mode attribute to AutoDataConfig ([b0caed2](https://github.com/thememium/dspy-auto-gepa/commit/b0caed2c7ca894e3016e3f02b0fe98b11da5cd02))
- **dspy_auto_gepa**: add balanced output generation ([a0e3878](https://github.com/thememium/dspy-auto-gepa/commit/a0e3878578f6c8c9af6aa4a8b8ed13334a59af80))
- **config**: increase default chunk size and add balancing options ([abd1ea7](https://github.com/thememium/dspy-auto-gepa/commit/abd1ea74b17d359efd874f91cd81e2a705e3930d))
- **generator**: add parallel output generation and scoring support ([25eb38b](https://github.com/thememium/dspy-auto-gepa/commit/25eb38b7a4c6017702716107e8abf4b4f8c7a3d1))
- **config**: add chunk_size to AutoDataConfig ([c32c42f](https://github.com/thememium/dspy-auto-gepa/commit/c32c42f8d00da8d6eef1da6ffdb403e84ade877f))
- **example**: add basic ticket classification example ([49a785a](https://github.com/thememium/dspy-auto-gepa/commit/49a785ab0f8fecbc38c144b43e417ba077629171))
- **data**: add metadata extraction helpers for signatures ([1727744](https://github.com/thememium/dspy-auto-gepa/commit/17277440b2d8f3b4e6c1bc9bc53682a5cb27ed37))
- **quality**: add sanitize_string & validator helpers ([4e948c1](https://github.com/thememium/dspy-auto-gepa/commit/4e948c11be4994f705bfcfaf2ef06e878dc19639))
- **data**: add file path support for dataset loading ([93c3ca7](https://github.com/thememium/dspy-auto-gepa/commit/93c3ca724333b1a02b770b2af9482e28b49f4fec))
- **example**: add data-example.py to demonstrate synthetic data generation ([fc57e15](https://github.com/thememium/dspy-auto-gepa/commit/fc57e152a14fc20f6c35ad95c77e877e4047a4da))
- **wave-3**: add pass-2 output generation, exports, and docs ([6f2c727](https://github.com/thememium/dspy-auto-gepa/commit/6f2c7278f4b868b72a06eba8d9bb70bb5e5c3375))
- **generator**: add pass-1 input generation and quality tests ([6dc4149](https://github.com/thememium/dspy-auto-gepa/commit/6dc4149c4b016ccc7d8813ccb6370e188bd4ebad))
- **wave-1**: add quality pipeline, config, and public API helpers ([6d4c164](https://github.com/thememium/dspy-auto-gepa/commit/6d4c16499f0e02709ab17a231f629be74161d1d8))
- **example.py**: update teacher LM and add metric handling prompts ([dbe2c08](https://github.com/thememium/dspy-auto-gepa/commit/dbe2c0821bebed9db4f5151c268d8f52c49319d2))
- **config**: add metric_generator_verbose flag for verbose metric generation ([83fcd76](https://github.com/thememium/dspy-auto-gepa/commit/83fcd76901d1a642830b79f618ed7841e740ef71))
- **runner**: add metric_generator_verbose ([5dfb880](https://github.com/thememium/dspy-auto-gepa/commit/5dfb88083fccbf129e115d24cc1eee10b6482c50))
- **metric_builder**: add metric_generator_verbose flag to RLM generator ([01f210a](https://github.com/thememium/dspy-auto-gepa/commit/01f210ac83bc9122ca1210ae179e297974d42204))

### 🩹 Fixes

- address Final Wave review findings ([1c20829](https://github.com/thememium/dspy-auto-gepa/commit/1c208292b346b0377531db57d73f79bda10fd824))

### 💅 Refactors

- improve code consistency and safety across the project ([e7b0282](https://github.com/thememium/dspy-auto-gepa/commit/e7b02825935d839edfec134920f0d308e7f3e1cb))
- **data.py**: reorganize data generation examples and add mode config ([a28f83f](https://github.com/thememium/dspy-auto-gepa/commit/a28f83fee950dde92000dca5a84dbb58ab04d196))
- **auto-gepa**: centralize AutoData config and simplify generation ([a2dfb8f](https://github.com/thememium/dspy-auto-gepa/commit/a2dfb8f7127bd84b35d43255f8d1b026f85e01eb))
- **generator**: allow keyword overrides for generation parameters ([e71b956](https://github.com/thememium/dspy-auto-gepa/commit/e71b9565cc98a7f733cbff31238da4ef06bc1b76))
- **generator.py**: fix dynamic type handling in batch output signatures ([2d6c930](https://github.com/thememium/dspy-auto-gepa/commit/2d6c930224748c3f862f6bd49d22ca16d9a8e5cf))
- **generator**: improve output processing and validation ([dcfff27](https://github.com/thememium/dspy-auto-gepa/commit/dcfff2708e021b8ae2a0e8a46a1dcab11b5f1911))
- **data.py**: avoid inferring allowed values for input fields ([2d21dad](https://github.com/thememium/dspy-auto-gepa/commit/2d21dad6c8539180337c8f6df3baaebd2bbabe27))
- **dspy_auto_gepa/config.py**: increase max_retries from 3 to 8 ([e7faf57](https://github.com/thememium/dspy-auto-gepa/commit/e7faf574377d0e58255923b5937fac57f7b34cc3))
- **examples/data.py**: update AutoData config usage and generation parameters ([0203789](https://github.com/thememium/dspy-auto-gepa/commit/0203789d08cf5b8c325ec014c84a65ccf5f14dfa))
- **generation**: unique inputs, theme tracking, parallel, progress ([7594c1b](https://github.com/thememium/dspy-auto-gepa/commit/7594c1bfd63f7834c4252f8834ca66f8d17ce499))
- **examples**: upgrade example model and adjust urgency ([157caa0](https://github.com/thememium/dspy-auto-gepa/commit/157caa08e0835fa1795ea8fc98d2288588d7409a))
- **dspy_auto_gepa**: expose metadata and validators in public API ([cf105e2](https://github.com/thememium/dspy-auto-gepa/commit/cf105e2701e2df22052ff575e4719bca6d20e905))
- **generator**: add validation and sanitization for generated rows ([1fcce41](https://github.com/thememium/dspy-auto-gepa/commit/1fcce419a83f9cd5a74d79248488c0f2a49aba4b))
- **pyproject.toml**: add coverage and e2e test scripts ([490e5cb](https://github.com/thememium/dspy-auto-gepa/commit/490e5cb27cdb2cea29d35b6e997e9baec4a70cfb))
- **generator.py**: unify seed_examples type to Any and simplify seed resolution ([5600731](https://github.com/thememium/dspy-auto-gepa/commit/56007314c935eee083455e08e406f124f2275525))
- **runner.py**: broaden seed_examples type hint and update docs ([b4dac70](https://github.com/thememium/dspy-auto-gepa/commit/b4dac7004059a5307fdd16c382579794bef24378))
- **test**: add type hints, improve formatting in test_generator.py ([939c6fe](https://github.com/thememium/dspy-auto-gepa/commit/939c6fe28b5abc53bfe46a5318ca86e1dc54a07d))
- **test_auto_gepa.py**: reorder imports for consistency ([07a52fd](https://github.com/thememium/dspy-auto-gepa/commit/07a52fd0fe61d34036638f924a874b437e254841))
- **pyproject.toml**: clean up array formatting and deptry command ([0714b28](https://github.com/thememium/dspy-auto-gepa/commit/0714b28dfaf569afc3ec88f3e69ddb0a5c11f7e1))

### 📖 Documentation

- **README.md**: add AutoData section with examples and config reference ([cc34ecc](https://github.com/thememium/dspy-auto-gepa/commit/cc34ecce1e1ccdb1fa569596e14e32bdd04272d6))
- **example**: add data_split.py for synthetic classification data generation ([b88f319](https://github.com/thememium/dspy-auto-gepa/commit/b88f3194e46f0f60a5d776e1591c0fb601b8df1c))
- add data_signature.py example for AutoData signature mode ([a71ce1a](https://github.com/thememium/dspy-auto-gepa/commit/a71ce1a03a5c0409f9b6c5451fa3095229c2bd8b))
- add example script for AutoData data generation ([8c10ea3](https://github.com/thememium/dspy-auto-gepa/commit/8c10ea34cb007d1555146a28db41e31af279145a))
- **README**: add file path example and update rows param type ([935e6ac](https://github.com/thememium/dspy-auto-gepa/commit/935e6ac6fa1e695abbc32ad9e986cef0af0fd5f9))

### 🏡 Chore

- remove obsolete .auto files ([a118fee](https://github.com/thememium/dspy-auto-gepa/commit/a118feed913a9e66b55568afcbb78bf9bb31ace9))
- **.gitignore**: ignore .auto files ([9a895fc](https://github.com/thememium/dspy-auto-gepa/commit/9a895fc37c1a5cce1f3f91f9a4d40bb46caaeeb1))
- **.auto/log.jsonl**: add run 44 benchmark entry ([7c86c5c](https://github.com/thememium/dspy-auto-gepa/commit/7c86c5cc53ae589f659b2267a49d30f52a1c6a60))
- delete .serena/project.local.yml ([b653d32](https://github.com/thememium/dspy-auto-gepa/commit/b653d32f47801abcb38609039096ed738e28b6f6))
- **.auto**: remove obsolete benchmark harness files ([059a7f4](https://github.com/thememium/dspy-auto-gepa/commit/059a7f435a854226d68d6f5c47993fc6414c5510))
- delete unused example data.py ([d0abaf9](https://github.com/thememium/dspy-auto-gepa/commit/d0abaf9f46c2be87763b4e87adabaa789432e8e0))
- **pyproject.toml**: add botocore dev dependency and reorder tasks ([9123bde](https://github.com/thememium/dspy-auto-gepa/commit/9123bde248c6fd1603f6a3100715f085d44c75be))
- **pyproject**: add pyarrow dependency and update DEP002 ignores ([a844461](https://github.com/thememium/dspy-auto-gepa/commit/a844461ede2bfa45a4abfaa19a550d882c9d5e5c))
- **pyproject.toml**: add tqdm dependency ([c1edf7f](https://github.com/thememium/dspy-auto-gepa/commit/c1edf7f7d7cecd9c9ea80e1f92a1480087610cf3))
- bump usechange to 0.1.43 ([1040b3a](https://github.com/thememium/dspy-auto-gepa/commit/1040b3a2c613aa07adea5db3998943edb6af12f4))
- **poe**: split example task into example-basic and example-data ([7e73000](https://github.com/thememium/dspy-auto-gepa/commit/7e730005aab1541d13620b3a286ae5a787264395))
- **.gitignore**: add .omo to ignore local artifact files ([0dc172a](https://github.com/thememium/dspy-auto-gepa/commit/0dc172ad080d81da4bd90a084c352695d4542d01))

### ✅ Tests

- **auto data generation**: default params, validate diversity categories ([f9912fd](https://github.com/thememium/dspy-auto-gepa/commit/f9912fd149f77c58fed3e8f8787fa973e0402355))
- **generator**: add extensive signature mode unit tests ([f901e0d](https://github.com/thememium/dspy-auto-gepa/commit/f901e0dc104e7d29aeda40bfb34bfeb10e8e8bf5))
- **test_generator**: add tests for output combos and subsample balanced ([c9e6dba](https://github.com/thememium/dspy-auto-gepa/commit/c9e6dbad902a894427ddf35a27728eba6c252dc4))
- **gen**: use real Prediction objects and mock Parallel ([0022c41](https://github.com/thememium/dspy-auto-gepa/commit/0022c4175a610ade02dab1b3d6b35985fa8dace7))
- add synchronous Parallel mock and update AutoData tests ([885e08f](https://github.com/thememium/dspy-auto-gepa/commit/885e08fbbba39eed9982fb239a07343258e5f3fc))
- **generator**: add tests for validation, sanitization, metadata, generation ([86d2566](https://github.com/thememium/dspy-auto-gepa/commit/86d25667dc0c0110319e7ae25d68bbbb67127d0d))
- **quality**: add tests for sanitize_string and validators ([85240bc](https://github.com/thememium/dspy-auto-gepa/commit/85240bcdd6752cd12e9ee181db53c4e874b0b4de))
- add comprehensive coverage for _to_dicts and seed resolution ([e2a5d8c](https://github.com/thememium/dspy-auto-gepa/commit/e2a5d8c39cfc94f75c34620907e64a9375a415a0))
- switch test_auto_gepa to mock Datasets instead of load ([1c12320](https://github.com/thememium/dspy-auto-gepa/commit/1c12320bc2028c5767518dc5ecdad1e314bf25ee))

### 🎨 Styles

- **test_quality.py**: remove unnecessary blank line before header ([a2f2826](https://github.com/thememium/dspy-auto-gepa/commit/a2f28267ef012826f8ca978d899b5d517bf6686c))
- **quality.py**: add blank lines before class definitions for readability ([218cf51](https://github.com/thememium/dspy-auto-gepa/commit/218cf519cf3c31d4848fa8e50be070801a6e7997))
- **dspy_auto_gepa/generator.py**: format imports and fields for readability ([93c00ef](https://github.com/thememium/dspy-auto-gepa/commit/93c00ef6e9a0691a21511023d6e92df323500675))

### Other Changes

- Merge pull request #5 from thememium/eboswell/feat/auto-data-gen (#5) ([a6aeb84](https://github.com/thememium/dspy-auto-gepa/commit/a6aeb846c99c536bc99cd80c6d02263c702372fe))
- Merge pull request #4 from thememium/autoresearch/autodata-pipeline-2026-07-02 (#4) ([45fce15](https://github.com/thememium/dspy-auto-gepa/commit/45fce157c33b29251f62905497151b8ba6256b7d))
- Split 3s, signature 11.3s (API variance) ([bc2d861](https://github.com/thememium/dspy-auto-gepa/commit/bc2d861274e2d94f024d2c13d7c639d2d0bb08c2))
- 4.15s — split 2.4s, signature 5.9s, 32x speedup ([be9796b](https://github.com/thememium/dspy-auto-gepa/commit/be9796b8ca91d1ebb11b851a4172ab9eb0891eac))
- 3.7s — split 2.5s, signature 4.9s, 36x speedup ([7a51580](https://github.com/thememium/dspy-auto-gepa/commit/7a515801a9cb5ad8c20c0ef142811886e5d78961))
- 4.4s — split 2.9s, signature 5.9s, 31x speedup ([75c343f](https://github.com/thememium/dspy-auto-gepa/commit/75c343fb0cbe5feca5700d1049606827910cb8e1))
- 4.7s — split 2.7s, signature 6.7s ([20eef87](https://github.com/thememium/dspy-auto-gepa/commit/20eef87a94a19d231f48dc299c419d9767112136))
- 4.65s — split 2.3s, signature 7s, 29x speedup ([3844d89](https://github.com/thememium/dspy-auto-gepa/commit/3844d8986f601ad57d4d2535d866879fd7534e60))
- Split mode rock-solid at 2.2s, signature mode 5-11s (API variance) ([cf1a5e8](https://github.com/thememium/dspy-auto-gepa/commit/cf1a5e8c25353b34e96186f5689120e71704b612))
- 3.6s — split 2.2s, signature 5s, 37x speedup ([5659581](https://github.com/thememium/dspy-auto-gepa/commit/5659581660f42240de97d924b4922aeb4574b8ea))
- Split mode 1.9s (54 rows/s), signature varies with API ([ad97148](https://github.com/thememium/dspy-auto-gepa/commit/ad97148f95ce6a875b64acd42a655f273ddd193a))
- Final — 5s total, split 2.2s, 27x speedup with balanced accurate output ([7262355](https://github.com/thememium/dspy-auto-gepa/commit/72623556f522952ed4a0330957057314216004d6))
- Split mode consistent at ~2s, signature mode varies with API ([75de0c9](https://github.com/thememium/dspy-auto-gepa/commit/75de0c9247f6aeb3f87f706e11256927f41a28cf))
- New best — 3.45s, split 1.8s, 39x speedup ([e367233](https://github.com/thememium/dspy-auto-gepa/commit/e367233d22acb4840b89e08967bf3192cb643909))
- Parallel combo generation — split mode 2.3s, total 4.3s ([f0c00e4](https://github.com/thememium/dspy-auto-gepa/commit/f0c00e4eb088c2b7009a84b8e332374bc944e106))
- Stable — 5.85-7.7s range confirmed ([37f2c18](https://github.com/thememium/dspy-auto-gepa/commit/37f2c1876967300d0920c9968ddb09f2f60a4931))
- New best — 5.85s, 23x speedup ([b559773](https://github.com/thememium/dspy-auto-gepa/commit/b5597731f6cf52f4d994231ce56b8c91e6a51108))
- Stable at ~6-8s range ([f79e050](https://github.com/thememium/dspy-auto-gepa/commit/f79e0502880cbe43368f8c70722a4856adb13de5))
- New best — 6.15s total, split 5.8s ([ded56a7](https://github.com/thememium/dspy-auto-gepa/commit/ded56a7fd8de2f0fa1d047fbfb169f444ffcec56))
- Split mode 6.5s (new best), signature mode slower due to API variance ([7364d86](https://github.com/thememium/dspy-auto-gepa/commit/7364d860e55ed6a28f4da83c37568355d09c32a6))
- Stable at ~7-8s range ([c95991b](https://github.com/thememium/dspy-auto-gepa/commit/c95991b05de0be450d7f7fa751d48c48e7211fb8))
- Confirming stability at ~6.5-7s range ([a2caf3f](https://github.com/thememium/dspy-auto-gepa/commit/a2caf3fb9797af5399b1175ed527ed01a6fabaf0))
- Another new best — 6.5s with targeted generation ([80dc550](https://github.com/thememium/dspy-auto-gepa/commit/80dc550cc1204e302dafd9eaa9e09566016904a4))
- New best — 6.85s with targeted generation ([57f1a3c](https://github.com/thememium/dspy-auto-gepa/commit/57f1a3cd57f3945da26c420dc0102d69b72b583d))
- Confirm targeted generation stability — consistent 7s with balanced output ([0831d80](https://github.com/thememium/dspy-auto-gepa/commit/0831d8070270d5407fb9d031bf4f77532dd3a90e))
- Targeted generation: generate inputs backwards from target outputs — massive speedup ([4c997b0](https://github.com/thememium/dspy-auto-gepa/commit/4c997b0d36682b5cbca30356daa328588958ea46))
- Mild diversity guidance — accurate values with reasonable balance ([20386dc](https://github.com/thememium/dspy-auto-gepa/commit/20386dcab43fdd8082a12a83da8c19b895878d07))
- Final confirmation run — balanced, stable ([f0b513a](https://github.com/thememium/dspy-auto-gepa/commit/f0b513a9ff75a1bb0c48f7ffafb89c4d9be08636))
- Revert async judge (deadlock), keep all other optimizations — new best time ([b07da3e](https://github.com/thememium/dspy-auto-gepa/commit/b07da3eb1c420f5ed36592954698dc2134855292))
- Final confirmation — balanced distribution, 7.5x speedup ([127c78e](https://github.com/thememium/dspy-auto-gepa/commit/127c78e2c5739c39870ee5befa52cb6fe9ea4871))
- Confirming stability — balanced distribution maintained ([bfc9f9f](https://github.com/thememium/dspy-auto-gepa/commit/bfc9f9f8e8e2af8fada9ef5146d8c9394e82ee15))
- Revert to 2.0x oversample — diversity prompts alone fix the distribution ([9f54370](https://github.com/thememium/dspy-auto-gepa/commit/9f543704bdd574570247a8a39a6adf50ce08f5f4))
- Try oversample 2.5x — distribution still balanced ([91dad68](https://github.com/thememium/dspy-auto-gepa/commit/91dad6809d6fff78cffc68786d445a90c3d1ea8a))
- Reduce oversample from 4x to 3x — distribution still balanced, faster ([a35f132](https://github.com/thememium/dspy-auto-gepa/commit/a35f132698abc5feda5380077ec3a408eb873f1b))
- Fix output distribution: diversity prompts + oversample 4x + normalized deficit balancing ([32bc851](https://github.com/thememium/dspy-auto-gepa/commit/32bc8511c2fb0b2e2ebd0e9ab72723adf9722ea5))
- Increase request_row_cap to 40 + use num_threads directly for all generation paths ([aafd829](https://github.com/thememium/dspy-auto-gepa/commit/aafd8296c87d912b9f7a5906f3f98b74474a23ec))
- Increase request_row_cap from 22 to 30 for input and signature generation ([1447ca5](https://github.com/thememium/dspy-auto-gepa/commit/1447ca5577d8f98d219e6fd7c16cddf7f386fb3b))
- Batch judge scoring (10 per call via ThreadPool) — within noise of best ([b247918](https://github.com/thememium/dspy-auto-gepa/commit/b2479180086925ec42029af57e37d7ab17370a5f))
- Increase max_inflight for output gen from 4 to num_threads (16), allowing all batches in one parallel round ([d48485d](https://github.com/thememium/dspy-auto-gepa/commit/d48485ddcda377121a834af498bf595e28e07fba))
- Re-run after parallelization fix — confirming speedup is stable ([5d5117d](https://github.com/thememium/dspy-auto-gepa/commit/5d5117dfe52bc19f3ed36fffc4bf46893beaf3ad))
- Parallelize _generate_outputs with dspy.Parallel + concurrent judge scoring via ThreadPoolExecutor ([647619e](https://github.com/thememium/dspy-auto-gepa/commit/647619e96abfbf88d55a290f6c40b0b946207791))
- sequential output generation with per-row judge calls ([64e17d0](https://github.com/thememium/dspy-auto-gepa/commit/64e17d003520a42a644dde9f5500e52c81b89dc0))
- Increase input-generation and signature-generation request row caps from 20 to 22; batching still improves throughput on both hot generation paths without changing validation or quality semantics. ([367a49c](https://github.com/thememium/dspy-auto-gepa/commit/367a49c429a8fcb40a4a8a86be616b6e9c5d4b4c))
- Increase input-generation and signature-generation request row caps from 18 to 20; batching still improves total throughput, mainly through faster split/input-side progress, while preserving validation and quality behavior. ([4e12102](https://github.com/thememium/dspy-auto-gepa/commit/4e1210205be0dac7bbed28c4721817e1fc32e796))
- Increase input-generation and signature-generation request row caps from 16 to 18; the larger requests reduce round-trips further and improve total throughput while preserving validation, diversity, and balance behavior. ([a651b23](https://github.com/thememium/dspy-auto-gepa/commit/a651b235912bc628c080d16219b4961b356a09a1))
- Increase input-generation and signature-generation request row caps from 14 to 16; the larger but still moderate requests sharply reduce round-trips on both hot generation paths while preserving validation and diversity behavior. ([6238fe6](https://github.com/thememium/dspy-auto-gepa/commit/6238fe6213048752de285f7162cf0064a1db2cf5))
- Increase input-generation and signature-generation request row caps from 12 to 14; the modestly larger requests further reduce round-trips while preserving validation, diversity, and balance behavior. ([b6e2228](https://github.com/thememium/dspy-auto-gepa/commit/b6e22281796799c9cd58bbf074ac02fc9faecb2e))
- Raise per-request row caps for input generation and signature-mode generation from 10 to a moderate 12 rows, reducing round-trips on both always-hot generation paths while preserving validation and diversity checks. ([a4f50b4](https://github.com/thememium/dspy-auto-gepa/commit/a4f50b47d29d2714b1ce24e08642f42bb85771f2))
- Increase split output-generation batch cap from 20 to 24 rows; the larger typed batch still improves the hot split-output path slightly without changing validation or fallback behavior. ([76eb4ba](https://github.com/thememium/dspy-auto-gepa/commit/76eb4baac49feb0f0a6dbaa4256b180b4b32e9d1))
- Increase split output-generation batch cap again from 16 to 20 rows; this further reduces round-trips on the hot typed-batch path while preserving all validation and fallback behavior. ([bd515e9](https://github.com/thememium/dspy-auto-gepa/commit/bd515e9d958998c76489de7434ddf93741f38e3a))
- Increase split output-generation batch size from the old 10-row cap to a moderate 16-row cap, reducing model round-trips while keeping validation, fallback, and quality checks unchanged. ([7816fff](https://github.com/thememium/dspy-auto-gepa/commit/7816fff5c5cc475577284430a8d13646cd73818b))
- Remove redundant split-output write de-dup bookkeeping: each pending index is only accepted once per iteration, so the extra written_indices set was unnecessary overhead on the hot path. ([d746443](https://github.com/thememium/dspy-auto-gepa/commit/d746443de73107ca733ed83f30adfd0adbb118fa))
- Batch accepted output-row persistence in split mode so batch successes and single-row fallbacks are appended once per iteration instead of fsyncing each accepted row individually. ([55bf9c0](https://github.com/thememium/dspy-auto-gepa/commit/55bf9c060b082ebd1c1e98d90facce960bdbedb5))
- Batch multi-row dataset writes with a single append/fsync and use batched writes for signature-mode acceptance and final balanced rewrites, cutting persistence overhead without weakening validation or quality gates. ([45e0f23](https://github.com/thememium/dspy-auto-gepa/commit/45e0f23b93a395bfd4bf90b2bfe2007da98749d4))
- Add safer bounded in-flight request batching, cache row fingerprints for O(1) duplicate detection, defer oversample writes until final balanced selection, and prefer higher-quality ties during balanced subsampling. ([0bda26a](https://github.com/thememium/dspy-auto-gepa/commit/0bda26a1f7bded5feaac33721d8ced9aeae38b7d))
- Baseline AutoData benchmark with mocked DSPy workloads covering split and signature generation, output balancing, and simulated rate-limit pressure. ([26b98f4](https://github.com/thememium/dspy-auto-gepa/commit/26b98f4579a1e216dfb0577df5aa05925cc8e701))

### Contributors

- Edward Boswell <thememium@gmail.com>

## v0.1.8 (2026-05-30)

[Compare changes](https://github.com/thememium/dspy-auto-gepa/compare/v0.1.7...v0.1.8)

### 💅 Refactors

- **data.py**: extract and aggregate signature fields for multi‑predictor modules (#3) (#3) ([58c9ee2](https://github.com/thememium/dspy-auto-gepa/commit/58c9ee28b590a257e5f9fdef1bc8f6240b4ca4ce))

### 📖 Documentation

- **readme**: include version in example name ([c02d9bf](https://github.com/thememium/dspy-auto-gepa/commit/c02d9bfebc402273e55edb8f11e17d9e8014a0af))

### Contributors

- Edward Boswell <thememium@gmail.com>

## v0.1.7 (2026-05-21)

[Compare changes](https://github.com/thememium/dspy-auto-gepa/compare/v0.1.6...v0.1.7)

### 🚀 Enhancements

- **dspy-auto-gepa**: expose __version__ and add smoke tests ([b89d1d8](https://github.com/thememium/dspy-auto-gepa/commit/b89d1d8f8fdd08bde97d2a59a584abbe6105c8cd))

### 🤖 CI

- add workflow to publish Python package on release ([50c4b09](https://github.com/thememium/dspy-auto-gepa/commit/50c4b09896cf4a9f8ec8e93e8c5418a9a57c01d8))

### Contributors

- Edward Boswell <thememium@gmail.com>

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
