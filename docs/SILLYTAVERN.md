# Using maya-csm with SillyTavern

## 1. Get a server URL

- **Colab/Kaggle:** run the notebook in `notebooks/`; it prints a
  `https://….trycloudflare.com` URL. The URL changes every session.
- **Local:** `python -m maya_csm --port 8000` → `http://localhost:8000`.

## 2. Configure the TTS extension

In SillyTavern: **Extensions (stacked-cubes icon) → TTS**.

| Setting | Value |
|---|---|
| Select TTS Provider | **OpenAI Compatible** |
| Provider Endpoint | `https://<your-url>/v1/audio/speech` |
| API Key | anything (e.g. `none`) — not checked |
| Model | `maya-csm` |
| Available Voices | `maya` |

Enable the extension, assign the `maya` voice to your character, and turn on
**Narrate by paragraphs (when not streaming)** if you use streaming responses.

With a Colab/Kaggle tunnel the URL must be updated here each new session.

If generation is slow (CPU mode), raise the request timeout: in
`config.yaml` of SillyTavern set `requestTimeout` higher, or prefer the GPU
notebooks.

## 3. Expressive tags in character output

Have the character (or your prompts) emit bracketed tags inline:

```
[giggling] Stop it, you're making me blush. [whispering] Come closer...
```

Supported tags and their behavior are listed in the project README. Tips:

- A tag applies until the next tag, so re-tag when the mood changes.
- Keep sentences short — long generations drift in voice quality.
- Ask the model for tags via your system prompt / character card, e.g.:
  > When expressing emotion, insert one of these inline tags before the
  > sentence: [laughing], [giggling], [whispering], [sighing], [gasping],
  > [crying], [excited], [nervous], [pause]. Use them sparingly.
- Asterisk actions like `*laughs*` are NOT translated (SillyTavern usually
  strips or narrates them); use square brackets.

## 4. Verify with curl

```bash
curl -X POST https://<your-url>/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"maya-csm","input":"[laughing] That is hilarious.","voice":"maya"}' \
  -o out.wav && xdg-open out.wav
```
