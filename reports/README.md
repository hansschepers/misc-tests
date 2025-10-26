# Multi-Project Reports Repository

This repository stores progress reports and analysis outputs from multiple projects.

## Structure

Each project has its own subdirectory under `reports/`:
- `reports/PROJECT_NAME/progress/` - Active reports (< 7 days old)
- `reports/PROJECT_NAME/archive/` - Archived reports (> 7 days old)
- `reports/PROJECT_NAME/summary_index.md` - Auto-generated index

## Adding a New Project
```bash
mkdir -p reports/YOUR_PROJECT_NAME/{progress,archive}
```

## Archiving

Reports are automatically archived weekly by GitHub Actions.
Manual trigger: Actions tab → "Archive Old Reports" → Run workflow
