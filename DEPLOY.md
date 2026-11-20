# Sharing the prototype safely

The input pack is Kuku's confidential content ("please don't publish or share it"), and the brief asks for links
that open for anyone who has them. So the public repository holds code only, and everything that quotes or
describes the show is shared by link:

| What | Where | Why |
|---|---|---|
| code, tests (synthetic show), README | public GitHub repo | holds no screenplay text, title or cast names |
| `raw_outputs/` (the run on the full pack) | Google Drive folder, "anyone with the link" | quotes hook and cut lines |
| the solution doc (PDF) | Google Drive, "anyone with the link" | quotes a few lines and names the cast |
| the review app | Streamlit Community Cloud, password-protected | shows the saved plan |

* The episode folder, `raw_outputs/`, `docs/`, `profiles/*.json`, the judges' answers in `eval/`, `.env`,
  `.llm_cache/`, `runs/` and `review/` are git-ignored.
* `python scripts/check_repo_safe.py` lists what would be committed and **stops** if a screenplay, a quoted output,
  the show's title or a cast name (learned from the episodes on your machine), a secrets file or an API key is in it.
  Run it before every push.

## Deploy on Streamlit Community Cloud (free)

1. `python scripts/check_repo_safe.py` must print `OK`; push to `main`.
2. Zip the outputs: `Compress-Archive raw_outputs raw_outputs.zip` (PowerShell) or `zip -r raw_outputs.zip
   raw_outputs`. Upload the zip to Google Drive and share it as "anyone with the link".
3. On https://share.streamlit.io choose *Create app*, the repository, branch `main`, file `app.py`, Python 3.12.
4. In *Advanced settings → Secrets* add:
   ```
   app_password = "<choose one>"
   raw_outputs_url = "<the Drive share link of raw_outputs.zip>"
   ```
5. Deploy, then put the app link and the password in the submission form.

The app downloads the zip once at start and shows the saved plan; the upload/paste runner works on any new
screenplay. To let reviewers try a model inside the app, add `GEMINI_API_KEY = "..."` to the same secrets box;
never type a key into a shared app.

## Never

* commit the episode folder, `raw_outputs/`, `docs/`, `.env` or `.llm_cache/`;
* paste an API key into the app on a shared deployment.
