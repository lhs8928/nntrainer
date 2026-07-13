# quick.ai Model Unit Tests

Reference guide for adding tests when integrating a new model family (e.g. LLaMA, Phi).

## Test Structure

```
test/unittest/models/
├── quick_ai_test_utils.h                      # Common interfaces, adapter template, helpers
├── quick_ai_test_utils.cpp                    # Helper implementations
├── unittest_quick_ai_qwen3.cpp                # Tiny deterministic tests (example)
├── unittest_quick_ai_qwen3_reference.cpp      # Differential tests (example)
└── quick_ai_reference/
    └── qwen3_tiny/                            # Committed HF reference fixture
        ├── config.json
        ├── generation_config.json
        ├── nntr_config.json
        ├── tokenizer.json
        ├── nntr_qwen3_tiny_fp32.bin
        ├── input_ids.json
        ├── reference_logits.json
        ├── reference_tokens.json
        └── meta.json
```

Tests are organized into two layers:

| Layer | File pattern | Description |
|-------|-------------|-------------|
| **Tiny deterministic** | `unittest_quick_ai_<model>.cpp` | Verifies golden logits with deterministic weights. No fixture required. |
| **Differential reference** | `unittest_quick_ai_<model>_reference.cpp` | Compares nntrainer output against a committed HF reference. Fixture required. |

---

## Key Types

### `CausalLMTestAdapter<ModelBase>`

A generic adapter template shared by all models. Inherits from `ModelBase`
(e.g. `Qwen3CausalLM`) and implements the `TinyCausalLMRunner` interface once,
so per-model test files only differ in how weights are populated.

```cpp
// Per-model using aliases (add to quick_ai_test_utils.h)
using Qwen3Adapter  = quick_ai_test::CausalLMTestAdapter<quick_ai::Qwen3CausalLM>;
using Qwen2Adapter  = quick_ai_test::CausalLMTestAdapter<quick_ai::Qwen2CausalLM>;
using Gemma3Adapter = quick_ai_test::CausalLMTestAdapter<quick_ai::Gemma3CausalLM>;
```

**Note**: If a model needs to pre-process its config before passing it to the
`Transformer` base constructor (e.g. Gemma3's `sanitizeConfig()`), add a thin
subclass that initializes `Transformer` with the processed config first.

### `TinyCausalLMCase`

Describes one tiny deterministic test case.

```cpp
struct TinyCausalLMCase {
  std::string name;                              // gtest parameter name
  TinyCausalLMDataType data_type;               // FP32 / Q4_0-FP32, etc.
  TinyCausalLMExpectedLogits expected_logits;   // golden logit vector + tolerance
  std::function<quick_ai::json()> make_model_config;
  std::function<std::map<...>(const TinyCausalLMDataType &)> make_layer_dtype_map;
  std::function<std::unique_ptr<TinyCausalLMRunner>(json &, json &, json &)> create_model;
  std::function<void(TinyCausalLMRunner &)> setup_weights; // populate deterministic weights
};
```

### `DifferentialModel`

Describes one differential test entry.

```cpp
struct DifferentialModel {
  std::string fixture_name;  // sub-directory under quick_ai_reference/ (e.g. "qwen3_tiny")
  std::function<std::unique_ptr<TinyCausalLMRunner>(json &, json &, json &)> make_model;
};
```

---

## Adding a New Model

### Step 1: Tiny deterministic tests (`unittest_quick_ai_<model>.cpp`)

#### 1-1. Add an adapter alias

Add to `quick_ai_test_utils.h`:

```cpp
using MyModelAdapter = quick_ai_test::CausalLMTestAdapter<quick_ai::MyModelCausalLM>;
```

#### 1-2. Write the deterministic weight setup function

```cpp
// unittest_quick_ai_mymodel.cpp
void setupMyModelDeterministicWeights(TinyMyModelCausalLM &model) {
  model.forEachLayer([](ml::train::Layer &layer,
                        nntrainer::RunLayerContext &ctx, void *) {
    for (unsigned int i = 0; i < ctx.getNumWeights(); ++i) {
      auto &w = ctx.getWeight(i);
      if (w.getDataType() != ml::train::TensorDim::DataType::FP32)
        continue;
      w.setValue(0.0f);
      if (layer.getType() == "rms_norm")
        w.setValue(1.0f);
      // Initialize embedding or other layers as needed
    }
  });
}
```

Weight visit order follows the **DFS topological sort** of the nntrainer graph
(not insertion order). For parallel branches where `wq`, `wk`, `wv` all consume
the same normed input, DFS visits them depth-first: `wq → q_norm → wk → k_norm → wv`.

#### 1-3. Write the `TinyCausalLMCase` factory

```cpp
quick_ai_test::TinyCausalLMCase
makeMyModelCase(const quick_ai_test::TinyCausalLMDataType &data_type) {
  return {
    "MyModel_" + data_type.name,
    data_type,
    {"hello tok4", makeExpectedMyModelLogits(),
     data_type.name == "FP32" ? 1e-4f : 1e-3f},
    makeTinyMyModelConfig,
    makeMyModelLayerDtypeMap,
    [](quick_ai::json &cfg, quick_ai::json &gen, quick_ai::json &nntr) {
      return std::make_unique<TinyMyModelCausalLM>(cfg, gen, nntr);
    },
    [](quick_ai_test::TinyCausalLMRunner &runner) {
      setupMyModelDeterministicWeights(
        static_cast<TinyMyModelCausalLM &>(runner));
    },
  };
}
```

#### 1-4. Register the parameterized test suite

```cpp
INSTANTIATE_TEST_SUITE_P(
  MyModel, CausalLMTinyModelTest,
  ::testing::Values(
    makeMyModelCase(quick_ai_test::makeTinyFp32DataType()),
    makeMyModelCase(quick_ai_test::makeTinyQ40Fp32DataType())),
  [](const ::testing::TestParamInfo<quick_ai_test::TinyCausalLMCase> &info) {
    return info.param.name;
  });
```

This automatically generates the following three tests for each variant (FP32 / Q4_0):
- `GreedyGenerationSelectsArgmaxLogit`
- `WeightRoundTripProducesSameLogits`
- `PromptProducesExpectedLogits`

### Step 2: Tiny config guidelines

The dimensions in `makeTiny<Model>Config()` must **exactly match** the C++ model
constructor. A mismatch causes a weight bin shape error at load time.

Recommended tiny dimensions:

| Field | Value |
|-------|-------|
| `hidden_size` | 64 |
| `intermediate_size` | 64 |
| `num_hidden_layers` | 1–2 |
| `num_attention_heads` | 8 |
| `num_key_value_heads` | 4 (GQA) |
| `head_dim` | 8 |
| `vocab_size` | 32 |
| `max_position_embeddings` | 8 |

The shared nntrainer config already sets `max_seq_len = 8`, `init_seq_len = 4`,
`num_to_generate = 1`.

**Critical**: The MHA core layer's `max_timestep` must be set to `MAX_SEQ_LEN`
(= `max_position_embeddings`). Setting it to `INIT_SEQ_LEN + NUM_TO_GENERATE`
causes an out-of-bounds access in the RoPE table during the first decode step.

### Step 3: Generate the differential reference fixture

#### 3-1. Write the generation script

Create
`test/unittest/models/quick_ai_reference/generators/generate_<model>_reference.py`.
Use the existing `generate_qwen3_reference.py` as a template.

Key points:
- `TINY_CONFIG` dimensions must exactly match C++ `makeTiny<Model>Config()`.
- **Reuse the production weight converter** (`res/<model>/weight_converter.py`).
  Do not write a separate conversion — reusing the converter validates it too.
  Import it via `REPO_ROOT / "Applications" / "quick_ai" / "res" / ...` (see the
  `CONVERTER_DIR` line in the qwen3 template).
- Use `input_ids = [1, 4, 2, 3]` and `n_gen = 4` as the standard test input.
- For models with `tie_word_embeddings=True`, do not save lm_head separately
  (`save_lm_head = not config.tie_word_embeddings`).
- For models with `query_pre_attn_scalar` (e.g. Gemma3), set it equal to `head_dim`
  so the HF attention scale matches the C++ MHA core (`1/sqrt(head_dim)`).

```bash
python3 test/unittest/models/quick_ai_reference/generators/generate_<model>_reference.py
```

Outputs:
- `config.json`, `generation_config.json`, `nntr_config.json`, `tokenizer.json`
- `nntr_<model>_tiny_fp32.bin`
- `input_ids.json`, `reference_logits.json`, `reference_tokens.json`
- `meta.json` — records tolerances and library versions

#### 3-2. Set tolerances in `meta.json`

`logits_atol_fp32` is determined by the HF ↔ nntrainer FP32 difference. For a
correct converter, `0.01` is sufficient.

`logits_atol_q40` accounts for Q4_0 quantization error. To calibrate it:
1. Start with a loose value (`5.0`) and verify the Q4_0 test passes.
2. Read the maximum deviation from the test output, then set the tolerance to
   that value multiplied by a safety factor (e.g. ×2).

```json
{
  "seed": 42,
  "n_gen": 4,
  "input_ids": [1, 4, 2, 3],
  "logits_atol_fp32": 0.01,
  "logits_atol_q40": 5.0,
  "prefix_match_min": 2
}
```

#### 3-3. Commit the fixture

```bash
git add test/unittest/models/quick_ai_reference/<model>_tiny/
git add test/unittest/models/quick_ai_reference/generators/generate_<model>_reference.py
```

Include the binary file (`.bin`). Tiny fixtures are approximately 100–200 KB,
which is negligible for the repository.

### Step 4: Write the differential reference test file

`unittest_quick_ai_<model>_reference.cpp` is intentionally minimal:

```cpp
// unittest_quick_ai_mymodel_reference.cpp

#include <quick_ai_test_utils.h>
#include <gtest/gtest.h>
#include <mymodel_causallm.h>
#include <memory>

namespace {

quick_ai_test::DifferentialModel myModelDescriptor() {
  return {
    "mymodel_tiny",
    [](quick_ai::json &cfg, quick_ai::json &gen, quick_ai::json &nntr) {
      return std::make_unique<
        quick_ai_test::CausalLMTestAdapter<quick_ai::MyModelCausalLM>>(
        cfg, gen, nntr);
    },
  };
}

TEST(MyModelDifferentialTest, FP32MatchesHFReference) {
  quick_ai_test::runFp32DifferentialChecks(myModelDescriptor());
}

TEST(MyModelDifferentialTest, Q40CloseToFP32Reference) {
  quick_ai_test::runQ40DifferentialChecks(myModelDescriptor());
}

} // namespace
```

Both tests skip automatically when the fixture is absent.

### Step 5: Register in `meson.build`

Add both source files to the `unittest_quick_ai_models` source list in
`Applications/quick_ai/meson.build`:

```meson
sources = [
  ...
  'unittest_quick_ai_mymodel.cpp',
  'unittest_quick_ai_mymodel_reference.cpp',
]
```

---

## Running the Tests

```bash
cd build
ninja unittest_quick_ai_models

# All tests (Q4_0 differential tests are SKIPPED without NNTR_QUANTIZE_BIN)
./Applications/quick_ai/unittest_quick_ai_models

# Include Q4_0 differential tests
NNTR_QUANTIZE_BIN=<path/to/nntr_quantize> \
  ./Applications/quick_ai/unittest_quick_ai_models

# One model only
./Applications/quick_ai/unittest_quick_ai_models --gtest_filter="*MyModel*"

# Differential tests only
./Applications/quick_ai/unittest_quick_ai_models --gtest_filter="*Differential*"
```

When `NNTR_QUANTIZE_BIN` is not set, `Q40CloseToFP32Reference` tests are reported
as SKIPPED, not FAILED.

To override the fixture root directory:
```bash
NNTRAINER_QUICK_AI_FIXTURE_DIR=/my/path ./unittest_quick_ai_models
```

---

## Checklist

- [ ] MHA core `max_timestep` is set to `MAX_SEQ_LEN`, not `INIT_SEQ_LEN + NUM_TO_GENERATE`
- [ ] Tiny config dimensions match between the C++ model and the Python generation script
- [ ] The production weight converter is reused (no separate conversion code)
- [ ] `lm_head` is not saved separately when `tie_word_embeddings = true`
- [ ] `query_pre_attn_scalar` (if applicable) is set equal to `head_dim`
- [ ] Weight converter saves tensors in DFS topological sort order
- [ ] Fixture is committed under `quick_ai_reference/<model>_tiny/`
- [ ] `logits_atol_fp32 ≤ 0.01` and the FP32 differential test passes
- [ ] Both source files are registered in `meson.build`
