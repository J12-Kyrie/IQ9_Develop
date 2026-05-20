# Qwen3-VL Profiling 测试说明

本文记录 `/mnt/workspace/develop/qwen3vl/tools/qwen3vl_profile_runner.py` 的使用方式、测试产物结构，以及 `summary.json` / `summary.csv` / process samples 中主要字段的含义。

## 运行环境

在目标板上运行，不使用系统 `/usr/bin/python3`，固定使用：

```bash
/home/ubuntu/miniforge3/bin/python
```

推荐部署目录：

```bash
/mnt/workspace/develop/qwen3vl
```

runner 路径：

```bash
/mnt/workspace/develop/qwen3vl/tools/qwen3vl_profile_runner.py
```

## Smoke Test

用于确认 runner、Genie profile 解析、Response 终端显示、采样 CSV 和基础产物是否正常。

```bash
cd /mnt/workspace/develop/qwen3vl

/home/ubuntu/miniforge3/bin/python tools/qwen3vl_profile_runner.py \
  --case-name dual_cdsp_baseline \
  --indices 14 \
  --repeats 1 \
  --warmup-index 14 \
  --sample-ms 100 \
  --timeout-sec 300 \
  --output-root /mnt/workspace/develop/qwen3vl/profiles \
  --skip-warmup
```

运行时终端会显示当前 run 的 Genie stdout/stderr，其中包含模型 Response。完整输出仍会保存到 `runs/*.stdout.txt`。

如果只想保存日志、不想在终端显示 Response，可以加：

```bash
--no-stream-response
```

## Baseline Test

默认基线测试为 1 次 warmup 加 4 个 idx 各 5 次正式测试：

```bash
cd /mnt/workspace/develop/qwen3vl

/home/ubuntu/miniforge3/bin/python tools/qwen3vl_profile_runner.py \
  --case-name dual_cdsp_baseline \
  --indices 0,7,14,19 \
  --repeats 5 \
  --warmup-index 14 \
  --sample-ms 100 \
  --timeout-sec 300 \
  --output-root /mnt/workspace/develop/qwen3vl/profiles
```

输出目录命名格式：

```text
/mnt/workspace/develop/qwen3vl/profiles/qwen3vl_<case-name>_<YYYYMMDD_HHMMSS>/
```

例如：

```text
/mnt/workspace/develop/qwen3vl/profiles/qwen3vl_dual_cdsp_baseline_20260430_091214/
```

## Profiling Cases

`--case-name dual_cdsp_baseline`

使用当前生产 `qwen3vl.json`，不修改配置。

`--case-name single_cdsp_device0`

在本次测试输出目录的 `config_snapshot/` 中生成 profiling 专用 JSON，只保留第一个 backend extension，并把所有 split 放到同一个 ctx-bin 组。不会覆盖生产 `qwen3vl.json`。

`--case-name swapped_cdsp_order`

在本次测试输出目录的 `config_snapshot/` 中生成 profiling 专用 JSON，保持 ctx-bin 分组不变，交换 backend extensions 顺序。不会覆盖生产 `qwen3vl.json`。

## 产物结构

每个测试目录包含：

```text
metadata.json
summary.json
summary.csv
pidstat.log
iostat.log
journal_qnn.log
config_snapshot/
  qwen3vl.json
  htp_backend_ext_config*.json
  run_qwen3vl.sh
runs/
  warmup_idx14.json
  warmup_idx14.stdout.txt
  run_idx<idx>_rep<rep>.json
  run_idx<idx>_rep<rep>.stdout.txt
samples/
  warmup_idx14_process_samples.csv
  run_idx<idx>_rep<rep>_process_samples.csv
```

失败 run 也会保留 stdout、return code、partial samples，并在 `summary.json` / `summary.csv` 中标记 `status=failed`。

## summary.csv 字段

`idx`

测试输入编号，对应 `test_data_onboard_1664/inputs_embeds_<idx>_uint16.bin`。

`rep`

重复次数编号，从 `0` 开始。warmup 不写入 `summary.csv`，只写入 `summary.json.warmup`。

`status`

`ok` 表示 Genie 命令返回 0 且 profile JSON 存在并可解析；`failed` 表示命令失败、profile 缺失或解析失败。

`returncode`

`sg fastrpc -c ... genie-t2t-run ...` 的进程返回码。

`profile_json`

本次 run 的 Genie `--profile` 原始 JSON 路径。

`stdout`

本次 run 的 Genie stdout/stderr 日志路径。终端实时显示的 Response 也会完整保存在这里。

`samples_csv`

本次 run 的 psutil 采样 CSV 路径。

`init_time_ms`

从 `GenieDialog_create.init-time` 解析出的初始化耗时，单位毫秒。

`query_duration_ms`

从 `GenieDialog_query.duration` 或同义字段解析出的 query 总耗时，单位毫秒。如果当前 Genie profile 没有该字段，则为空。

`prompt_tokens`

prompt token 数量，来自 `num-prompt-tokens`。

`prompt_processing_rate`

prompt processing rate，单位 tok/s。

`ttft_ms`

time to first token，单位毫秒。

`generated_tokens`

生成 token 数量，来自 `num-generated-tokens`。

`token_generation_time_ms`

token generation time，单位毫秒。

`token_generation_rate`

生成速度，单位 tok/s。该字段是性能回归判断的主要指标。

`ssd_acceptance_rate`

SSD speculative decoding 的 token acceptance rate。非 SSD 或 profile 无该字段时为空。

`peak_rss_bytes`

本次 run 采样到的最大 RSS，单位 bytes。

`peak_pss_bytes`

本次 run 采样到的最大 PSS，单位 bytes。若系统或权限无法读取 PSS，则回退到 RSS。

`regression_candidate`

单次 `token_generation_rate < 22 tok/s` 时为 `true`。

`error`

失败原因。成功 run 为空字符串。

## summary.json 字段

`warmup`

warmup run 的完整记录，不进入统计。

`runs`

正式测试每次 run 的完整记录，字段与 `summary.csv` 基本一致，并额外包含 `command` 和 `sampled_process_names`。

`per_idx`

按 idx 聚合后的统计，每个指标包含：

```text
mean
median
min
max
std
```

其中 `per_idx.<idx>.regression_candidate=true` 表示该 idx 的平均 `token_generation_rate < 23 tok/s`。

`overall`

全部成功 run 的总体统计：

```text
run_count
ok_count
failed_count
generation_rate
ttft_ms
prompt_processing_rate
peak_rss_bytes
peak_pss_bytes
regression_candidate
```

其中 `overall.regression_candidate=true` 表示全部成功 run 的平均 `token_generation_rate < 23 tok/s`。

## samples CSV 字段

每个 run 都有独立采样文件：

```text
samples/run_idx<idx>_rep<rep>_process_samples.csv
```

采样目标包含 runner 启动的 `sg` / shell / `timeout` / `genie-t2t-run` 进程树，以及 DSP daemon：

```text
cdsprpcd
adsprpcd
```

字段含义：

`timestamp_iso`

采样时间，设备本地时区 ISO 格式。

`epoch`

采样时间，Unix epoch seconds。

`kind`

`run_tree` 表示本次 Genie 命令进程树；`dsp_daemon` 表示常驻 DSP daemon。

`pid`

进程 ID。

`name`

进程名。

`cmdline`

进程命令行。

`cpu_percent`

psutil 采样到的 CPU 使用率。

`rss_bytes`

RSS，单位 bytes。

`pss_bytes`

PSS，单位 bytes；不可用时回退为 RSS。

`uss_bytes`

USS，单位 bytes；不可用时为空。

`num_threads`

线程数。

`read_bytes`

进程累计读取字节数；不可用时为空。

`write_bytes`

进程累计写入字节数；不可用时为空。

## 已验证结果

2026-04-30 在目标板执行过完整 baseline：

```text
/mnt/workspace/develop/qwen3vl/profiles/qwen3vl_dual_cdsp_baseline_20260430_091214
```

结果：

```text
summary.csv rows: 20
ok_count: 20
failed_count: 0
mean generation rate: 24.36 tok/s
mean TTFT: 1237.28 ms
mean prompt processing rate: 1345.63 tok/s
overall regression_candidate: false
sampled process names: adsprpcd, cdsprpcd, genie-t2t-run, sg, sh, timeout
```
