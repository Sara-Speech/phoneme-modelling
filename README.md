# phoneme-modelling

Generates synthetic speech sounds, distorts them into graded pronunciation errors, and evaluates the results.

```bash
make install-dev   # install everything, including test tools
make generate      # generate clean clips, then distort them
make eval          # score the clips
make run           # both
```

Settings live in `config/default.json`.

`make eval` reads `data/phonemes_metadata.csv`. To evaluate the hand-reviewed clips downloaded from Drive (the audio in
`data/phonemes/` plus `data/phonemes_metadata.xlsx`), first copy the spreadsheet to that CSV. Rows marked Y in its
`Remove?` column are skipped. Rerun this whenever the spreadsheet changes:

```bash
PYTHONPATH=src uv run python -m phoneme_modelling.util.prepare_metadata
```

The `make` targets run `src/phoneme_modelling/generate_synthetic_audio.py` with default options (`make eval` is
`--mode eval`). Run the script directly to change them for one run. Start the command with `PYTHONPATH=src`, as the
Makefile does: on synced folders macOS hides the package's `.pth` file in `.venv`, and Python then can't find
`phoneme_modelling`.

| Option | Default | What it sets |
|---|---|---|
| `--mode` | `all` | `generate`, `eval`, or `all` (both) |
| `--evaluators` | all in the config | Only these entries of `evaluator_configs` |
| `--config_path` | `config/default.json` | Settings file |
| `--output_metadata_path` | `data/phonemes_metadata.csv` | Clip list: written by generate, read by eval |
| `--evaluation_output_path` | `data/evaluation_results.csv` | Per-clip results |
| `--evaluation_summary_path` | `data/evaluation_summary.csv` | Accuracy tables |

For example, to run only the Gemini 4 listeners and keep their results separate:

```bash
PYTHONPATH=src uv run python src/phoneme_modelling/generate_synthetic_audio.py --mode eval \
  --evaluators gemini_4_neon_listener gemini_4_carbon_listener \
  --evaluation_output_path data/gemini4_results.csv \
  --evaluation_summary_path data/gemini4_summary.csv
```

The LLM listeners cache each answer in `data/llm_listener_cache/`, so rerunning them only pays for clips they haven't
answered yet.

## Required Setup

1. **Google Cloud Platform (GCP) Authentication & APIs:**
   * Authenticate your local environment using the Google Cloud CLI:
     ```bash
     gcloud auth application-default login
     ```
   * Ensure you have a GCP project to serve as your quota project.
   * Enable the **Agent Platform API** and the **Cloud Text-to-Speech API** in your GCP project.

2. **AWS SSO Setup:**
   * Run AWS SSO configuration:
     ```bash
     aws configure sso
     ```
    * SSO session name: sara
    * SSO start URL: https://identitycenter.amazonaws.com/ssoins-668464a2f6d7a56e
    * SSO region: us-east-2
    * SSO registration scopes: sso:account:access
    * Default client Region: us-east-2
    * CLI default output format (json if not specified): None
    * Configure AWS skills and the AWS MCP server for your AI coding agent(s)?: n
   * Save the profile name returned for the AWS_PROFILE env variable in step 3

3. **Environment Variables:**
   * Retrieve the OpenAI API key from the AWS Secrets Manager: [openai_content_gen](https://381492152880-dopmmgh3.us-east-2.console.aws.amazon.com/secretsmanager/secret?name=openai_content_gen&region=us-east-2&tab=overview#).
   * Create a `.env` file in the root of the project with your OpenAI API key and GCP project ID:
     ```env
     OPENAI_API_KEY=your_retrieved_aws_secret
     GCP_PROJECT=your_gcp_project_id
     AWS_PROFILE=your_profile_from_step_2
     ```

## Distortions

A distortion edits a clean recording of a correct sound until it sounds like a common mistake, such as "r" said as
"w". Each sound pair is one class in `src/phoneme_modelling/distortion/distorters/`, because each pair needs its own
edit. `PairDistorter` does what they all share: read the clip, find the sound, apply the pair's edit at the requested
strength, and save a 16 kHz clip of at least 2.5 seconds.

Every pair has two versions:

- **Plain** (`r_to_w`): the distortion alone.
- **Matched** (`r_to_w_matched`): the same distortion, with the length, pitch and loudness of a correct clip of the
  same sound, so a model can't use those to tell errors from correct clips.

Each distortion is made at three strengths, set by `distortion_application_config.distortion_intensitites`:

- **0.3** goes about 30% of the way from the correct sound to the mistake: a slight version of the mistake.
- **0.6** goes about 60% of the way.
- **1** goes all the way.

Every strength is labelled the same way: `actual_phoneme_text` is the mistake (e.g. w), `distortion_phoneme_text`
is the correct sound (r), and `distortion_intensity` says how far the edit went (0.3, 0.6 or 1).

Any other strength is skipped. There is no strength 0: an unedited copy would only duplicate the clean clip, and the
clean clips are already the "correct" examples.

The numbers each pair uses are saved in `config/distortion/`, one file per pair. They were measured on the 2026-10-25
clean clips by `scripts/export_pair_parameters.py` in `phoneme-endpoint`. The 0.6 and 1 settings are that study's
severity 3 and 5. The 0.3 setting is read off the same calibration curve, at 30% of the way.

### Pairs

There are 57 pairs, so 114 distorters. They are grouped below by the edit, which is a function in
`distortion/recipes.py`.

| Edit | Pairs |
|---|---|
| Move formants (`shift_formants`) | r→w, r→l, r→y, distorted r (`r_distortion`, part way to w), l→w, l→y, y→w, w→r |
| Reshape the hiss and add voicing (`voice_fricative`) | s→z, s→voiced th, f→v |
| Remove voicing and reshape the hiss (`devoice_fricative`) | z→voiceless th, z→s, voiced th→voiceless th, v→f |
| Reshape the hiss only (`reshape_fricative`) | s→voiceless th, z→voiced th, voiced th→v, voiceless th→f, voiceless th→s, sh→s |
| Turn a hiss into a stop or ch (`fricative_to_stop`) | s→d, s→t, z→d, z→t, voiced th→d, voiceless th→t, sh→t, sh→ch, v→b, f→p, f→b |
| Turn ch or j into a hiss (`affricate_to_fricative`) | ch→sh, j→z, j→sh |
| Turn ch or j into a stop (`affricate_to_stop`) | ch→t, ch→k, j→d |
| Change a stop's voicing or place (`change_stop`) | k→t, k→g, g→d, g→k, t→k, t→d, t→g, d→t, d→k, d→g, p→b, b→p |
| Reshape the nasal hum (`change_nasal_place`) | ng→n, n→m, m→n |
| Turn a nasal into a stop (`nasal_to_stop`) | ng→g, n→t, n→d, m→b |

Each pair's docstring says what its edit does. It also flags weak pairs: those the 10-25 scorecard did not rate
"works", and those measured on fewer than 3 voices.

The endpoint lists 8 more mistakes that have no distorter: r→h, r→d, r→n, r→z, g→w, g→y, t→voiceless th and m→w. The
two sounds in each differ too much for an edit to move one toward the other. On 2026-10-25 they were covered only by
real recordings of the other sound, and the clean clips here already provide those.

### Adding a pair

1. Make sure `scripts/export_pair_parameters.py` exports the pair (it exports every pair with a graded edit), and run
   it to create the pair's settings file.
2. Add `distortion/distorters/<from>_to_<to>.py` with a plain and a matched class. `_apply` is the pair's edit, usually
   one call to a function in `distortion/recipes.py`.
3. Add both classes to `PAIR_DISTORTERS` in `distortion/pair_registry.py`, and add their names to `distorter_configs`
   in the config.

## Google Cloud TTS generator

`google_tts` (in `generator_configs`) speaks each sound with an SSML IPA tag inside "Please say ___ again.", finds the
sound with forced alignment (wav2vec2 espeak), crops it and preps it like the other generators. Continuants (r, l, w, y,
s, z, sh, f, v, h, m, n, ng) are held by repeating their symbol inside the tag (`hold_repeats`, default 8, about 0.3 to
0.7 s). Stops, ch, j and the two th sounds are short single clips (ch and j get a longer fricative half), so they are
made once per voice. Several values in `hold_repeats` make numbered takes (`..._take1`, `..._take2`) for held sounds.

- **Credentials:** Application Default Credentials for a project with the Cloud Text-to-Speech API enabled
  (`gcloud auth application-default login`, or `auth.authenticate_user()` in Colab). `project` is the quota project.
- **Voices:** any Cloud TTS voice name that supports IPA tags (Standard, Wavenet, Neural2). The voice name in a clip's
  name is the short form, e.g. `neural2f`.
- **Distortions:** this generator makes correct clips only. Add the new voices' pitch to `config/distortion/voice_f0.json`
  so the pair distorters use the right F0 for them (otherwise they fall back to the default).
- **Tests:** `cd src && uv run pytest ../tests/synthetic_audio`

## Tests

```bash
cd src && uv run pytest ../tests/distortion
```
