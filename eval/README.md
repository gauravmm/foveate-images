# Autonomous calendar evaluation

The evaluation gives each fresh Pi/Qwen agent only:

- one synthetic calendar copied to an isolated directory as `input.png`;
- the `foveate-images` skill;
- `read`, `bash`, `write`, and `ls` tools; and
- [calendar_prompt.md](calendar_prompt.md).

Ground-truth sidecars and earlier pipeline artifacts are not present in the agent directory. Each agent must choose its own grid, run `foveate`, inspect the resulting images, and write `answer.json`.

Qwen image reasoning is disabled with both Pi's `--thinking off` and:

``` json
{
  "compat": {"thinkingFormat": "qwen-chat-template"}
}
```

This causes Pi to send `chat_template_kwargs: {"enable_thinking": false, "preserve_thinking": true}`. Reject a trial if its completed assistant messages contain any `thinking` content blocks.

Score answers and summarize their Pi traces with:

``` bash
python scripts/score_calendar_results.py \
  runs/synthetic-calendars/generated \
  runs/qwen3.8-27b-foveate \
  --output runs/qwen3.8-27b-foveate/report.json
```
