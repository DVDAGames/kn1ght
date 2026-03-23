# Contributing

## Development Setup

```bash
uv sync
uv run nbstripout --install
```

`nbstripout` is a git filter that automatically strips outputs and execution counts from Jupyter notebooks before committing. Your local notebooks retain their outputs — only the committed blob is cleaned.

If you skip this step, git will warn you when staging `.ipynb` files.
