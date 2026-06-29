# Migration Report: ST Edge AI DC Pipeline Integration

## Executive Summary

Our current `stedge_compile.py` stage has **critical bugs** that prevent it from working with the official `stm32ai_dc` SDK. The `CloudBackend` constructor, method calls, and environment variable names are all wrong. This report details every issue and the required fix.

---

## What We Are Doing Correctly

1. **Environment variable for credentials** — We correctly use env vars (`STEDGE_USERNAME`/`STEDGE_PASSWORD`) instead of hardcoding, though the var names need updating.
2. **Stage architecture** — Our `Stage` base class, `should_skip()`, `validate_inputs()`, and `run()` pattern is clean and compatible.
3. **Graceful import fallback** — We check for `stm32ai_dc` ImportError and provide install instructions.
4. **Config-driven** — Target, credentials, and output directory are all configurable.

---

## Critical Issues Found

### Issue 1: CloudBackend Constructor Signature Mismatch

**Our code (BROKEN):**
```python
ai = Stm32Ai(
    CloudBackend(
        str(ctx.tflite_model),  # ← This is a FILE PATH
        username=username,
        password=password,
    )
)
```

**Official SDK:**
```python
CloudBackend(
    username: str,
    password: str,
    version: Union[str, None] = None,
    platform: BackendVersionType = BackendVersionType.STM32,
    silent = False,
)
```

**Impact:** `CloudBackend` interprets our file path as the `username` parameter. Authentication will fail immediately.

**Fix:** Pass `username` and `password` as positional args. No file path.

---

### Issue 2: Missing Model Upload Step

**Our code:** Calls `ai.compile()` directly on a local file path.

**Official SDK flow:**
1. `ai.upload_model(filePath)` — uploads to cloud, returns bool
2. `ai.generate(CliParameters(model=filename))` — generates C code
3. Or `ai.generate_nbg(model_name)` — generates NBG for MPU

**Impact:** The SDK operates on models already uploaded to the cloud. We must upload first.

---

### Issue 3: Wrong Method Name (`compile` vs `generate`/`generate_nbg`)

**Our code:**
```python
ai.compile(target=cfg.target)
```

**Official SDK:** No `compile()` method exists. The correct methods are:
- `generate(options: CliParameters)` — for C-code generation (MCU)
- `generate_nbg(model_name, timeout)` — for NBG generation (MPU)

**Impact:** `AttributeError` at runtime. For STM32MP1 (MPU), we need `generate_nbg()`.

---

### Issue 4: Wrong Download Method

**Our code:**
```python
ai.download(output_path=str(nb_path))
```

**Official SDK:**
```python
ai.download_model(model_name: str, path: str)
```

**Impact:** `AttributeError`. The SDK downloads by model name, not by output path.

---

### Issue 5: Environment Variable Name Mismatch

**Our code:** `STEDGE_USERNAME`, `STEDGE_PASSWORD`

**Official SDK README:** `STM32AI_USERNAME`, `STM32AI_PASSWORD`

**Impact:** Credentials won't be found if user follows SDK docs.

---

### Issue 6: No CliParameters Usage

**Our code:** Calls methods without proper `CliParameters` typed tuple.

**Official SDK:** All methods require `CliParameters(model=filename, type=CliParameterType.TFLITE, ...)`.

**Impact:** Even if other issues were fixed, the SDK won't know the model type or configuration.

---

### Issue 7: No Error Handling with SDK Exceptions

**Our code:** Catches generic `Exception`.

**Official SDK:** Raises specific exceptions (`InvalidCrendetialsException`, `GenerateNbgFailure`, `LoginFailureException`, etc.).

**Impact:** Poor error messages, inability to retry on transient failures.

---

### Issue 8: No Retry Logic

**Our code:** No retries on API calls.

**Official SDK:** `LoginService` retries login 3 times. `send_get` uses `Retry(total=5, backoff_factor=0.5)`. But `generate_nbg` polling has no retry on transient HTTP errors.

**Impact:** Transient network issues cause pipeline failure.

---

## Refactoring Plan

### Before → After

#### Authentication
```python
# BEFORE (broken)
ai = Stm32Ai(
    CloudBackend(
        str(ctx.tflite_model),
        username=username,
        password=password,
    )
)

# AFTER (correct)
from stm32ai_dc import Stm32Ai, CloudBackend
backend = CloudBackend(
    username=username,
    password=password,
    platform=BackendVersionType.STM32MPU,
)
ai = Stm32Ai(backend)
```

#### Upload → Analyze → NBG → Download
```python
# BEFORE (broken)
ai.compile(target=cfg.target)
ai.download(output_path=str(nb_path))

# AFTER (correct - MPU flow)
# 1. Upload
ai.upload_model(str(ctx.tflite_model))

# 2. Generate NBG for MPU
blob_name = ai.generate_nbg(
    model_name=ctx.tflite_model.name,
    timeout=600,
)

# 3. Download
ai.download_model(blob_name, str(output_dir))
```

#### Error Handling
```python
# BEFORE (generic)
except Exception as exc:
    return StageResult(status=StageStatus.FAILED, message=str(exc))

# AFTER (specific)
from stm32ai_dc.errors import (
    InvalidCrendetialsException,
    LoginFailureException,
    GenerateNbgFailure,
)
try:
    ...
except InvalidCrendetialsException:
    return StageResult(status=StageStatus.FAILED,
        message="Invalid ST Edge AI credentials. Check STEDGE_USERNAME/PASSWORD.")
except LoginFailureException as e:
    return StageResult(status=StageStatus.FAILED,
        message=f"Login failed: {e}")
except GenerateNbgFailure as e:
    return StageResult(status=StageStatus.FAILED,
        message=f"NBG generation failed: {e}")
```

---

## Risk Assessment

| Change | Risk | Mitigation |
|--------|------|------------|
| Fix CloudBackend constructor | Low — straightforward fix | Test with real credentials |
| Add upload step | Low — SDK handles it | Verify file exists before upload |
| Switch compile → generate_nbg | Medium — different API shape | Test with small model first |
| Fix download method | Low — straightforward fix | Verify model name from upload |
| Update env var names | Low — add fallback for old names | Support both STEDGE_* and STM32AI_* |
| Add retry with backoff | Low — additive change | Use exponential backoff, max 3 retries |
| Add SDK-specific error handling | Low — additive change | Catch known exceptions, log details |

---

## Backward Compatibility

- Support both `STEDGE_USERNAME`/`STEDGE_PASSWORD` and `STM32AI_USERNAME`/`STM32AI_PASSWORD` env vars
- Keep `stedge.enabled` config flag
- Keep `stedge.target` config (maps to `CliParameters(series=...)`)
- Output artifact (`nb_model`) stays in same location
