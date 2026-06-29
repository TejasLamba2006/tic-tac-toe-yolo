# STM32AI DC SDK Analysis

Cloned from: `https://github.com/STMicroelectronics/stm32ai-modelzoo-services.git`
Path: `common/stm32ai_dc/`

## API Methods (Stm32Ai class)

| Method | Signature | Returns |
|--------|-----------|---------|
| `analyze` | `(options: CliParameters)` | `AnalyzeResult` |
| `generate` | `(options: CliParameters, timeout=300)` | `GenerateResult` |
| `validate` | `(options: CliParameters)` | `ValidateResult` |
| `quantize` | `(options: CliParameters)` | any |
| `benchmark` | `(options: CliParameters, board_name: str, timeout=600)` | `MpuBenchmarkResult \| BenchmarkResult` |
| `generate_nbg` | `(model_name, timeout=300)` | `str` (blobName) |
| `get_benchmark_boards` | `()` | `List[BoardData]` |
| `get_user` | `()` | any |
| `list_models` | `()` | list |
| `upload_model` | `(filePath: str)` | bool |
| `download_model` | `(model_name, path)` | None |
| `delete_model` | `(modelName: str)` | bool |
| `upload_validation_input_file` | `(filePath: str)` | bool |
| `upload_validation_output_file` | `(filePath: str)` | bool |
| `list_validation_input_files` | `()` | list |
| `list_validation_output_files` | `()` | list |
| `list_generated_files` | `()` | list |
| `download_generated_file` | `(filePath)` | str |

## Authentication Flow

1. `CloudBackend.__init__(username, password, version, platform)`
2. Calls `LoginService.login(username, password)` which:
   - Opens SSO URL (`https://sso.st.com/as/authorization.oauth2`)
   - Follows redirects to my.st.com CAS login
   - POSTs credentials to CAS form
   - Follows redirects back to callback URL
   - Exchanges auth code for JWT token via `/login/callback`
   - Saves token to `~/.stmai_token`
3. Token refresh via `/login/refresh` with refresh_token
4. All API calls use `Authorization: Bearer {token}` header

## File Upload/Download

- **Upload**: `FileService.upload_model(filePath)` → POST multipart to `/api/file/files/models`
- **Download**: `FileService.download_model(model_name, path)` → GET with stream to local path
- Models are referenced by filename (not path) after upload

## generate_nbg() Flow (MPU)

1. Model must already be uploaded via `upload_model()`
2. Call `generate_nbg_service.trigger_optimize(model_name)` → POST to `/api/generate_nbg/optimize`
3. Returns `runtimeId`
4. Poll `generate_nbg_service.wait_for_run(runtime_id, timeout)` → GET `/api/generate_nbg/run/{id}`
5. States: `in_queue` → `done` | `error`
6. On success, returns `blobName` (the optimized model name)

## Error Handling

- `InvalidCrendetialsException` — bad login
- `BlockedAccountException` — too many failed logins
- `LoginFailureException` — general auth failure
- `GenerateNbgFailure` — NBG generation failed
- `AnalyzeServerError`, `GenerateServerError`, `ValidateServerError` — server errors
- `BenchmarkServerError`, `BenchmarkFailure` — benchmark errors
- `ServerError` — base class for server errors
- `ModelNotFoundError`, `ParameterError`, `FileFormatError` — client errors

## CliParameters Mapping

`CliParameters` is a `NamedTuple` with:
- `model` (required) — filename of uploaded model
- `type` — `CliParameterType` enum (KERAS, TFLITE, ONNX)
- `target` — target board/serie string
- `series` — MCU series (e.g., 'stm32f4')
- `optimization` — `CliParameterOptimization` enum (BALANCED, TIME, RAM)
- `compression` — `CliParameterCompression` enum
- Many more optional fields for advanced configuration

## MpuParameters (for MPU targets)

- `model` (required) — filename
- `engine` — `MpuEngine` enum (CPU, HW_ACCELERATOR)
- `nbCores` — int (default 1)

## Cross-Reference: What Our Pipeline Needs

| Feature | SDK Support | Status |
|---------|-------------|--------|
| Model upload (.tflite, .onnx) | Yes | We need to add upload step |
| NBG generation for STM32MP1 | Yes (`generate_nbg()`) | We're calling wrong method |
| Benchmarking | Yes (`benchmark()`) | Supported but different API |
| Validation with I/O datasets | Yes (`validate()`) | Supported |
| C-code generation | Yes (`generate()`) | For MCU, not MPU |
