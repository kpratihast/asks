# Import student and professor documents

Run the command from this repository. On the configured Mac it is also available as `/Users/kshitij/Desktop/askiisc/ingest-document.sh`.

```bash
# Student-only material; professors can also access it.
./ingest-document.sh --file /absolute/path/student-notice.pdf \
  --role student --title "Laboratory orientation" \
  --topic "laboratory orientation"

# Professor-only material.
./ingest-document.sh --file /absolute/path/exam-instructions.pdf \
  --role prof --title "Examination instructions" \
  --doc-type policy --term 2026-AUG --topic "examination instructions"
```

`prof`, `professor` and `faculty` map to the server's `faculty` scope. `student` permits students and professors. There is deliberately no public/default role in this private-document command. Public crawling is a separate workflow.

Supported originals: PDF, UTF-8 TXT/Markdown/HTML, PNG, JPG/JPEG and WebP. Text PDFs are extracted offline in a bounded child process; no file is sent to a model or OCR service. Images and scanned/empty PDFs require a reviewed transcription:

```bash
./ingest-document.sh --file /absolute/path/course-schedule.png \
  --role student --id course-schedule-2027 \
  --title "Course schedule Jan-Apr 2027" --term 2027-JAN \
  --doc-type schedule --extracted /absolute/path/reviewed-schedule.html \
  --topic "course schedule" --topic "class timings"
```

For schedules, retain table rows, column headings, course IDs and the term in reviewed HTML so graph relationships can be extracted. Plain-text ingestion does not create arbitrary new graph relation types: it uses the existing rule extractors. Review the extracted file printed by the command before promoting its index, especially dates, table columns and OCR results.

The importer:

1. Validates the role, ID, file sizes, extraction and existing catalog checksums. Clear instruction overrides in the extracted text are rejected.
2. Copies immutable originals/extractions to `.local/private/` with private file permissions and computes their SHA-256 checksums.
3. Clones the configured index into a fresh directory, preserves public content and admissions caches, reimports the classified documents, and rebuilds the graph and snapshot.
4. Checks the new document produced chunks with the intended scope. Only then does it atomically update `config/restricted_documents.json`.
5. Prints the new index path, extraction review path, and exact local-test/AWS-deploy commands. It does not change `config/local.json`, restart the current local app, or deploy automatically.

Failed imports do not register a document or alter the base index. An operator lock serializes imports; concurrent manual edits to the manifest are detected. Interrupted processes may leave unused files, but they do not publish a partially written catalog.

## Test, then deploy

Use the exact commands printed by the importer. For example, from the repository root:

```bash
ASKIISC_DATA_DIR="/absolute/path/to/new-index" ./run-local.sh --port 8102

./deploy-aws.sh --build --data-dir "/absolute/path/to/new-index"
```

Test a public question without login, the new document after a student login, and professor access. A professor document must remain denied to the student. When finished testing, stop the extra local server normally. To make the new index your local default, update `ASKIISC_DATA_DIR` in `config/local.json` and restart your usual local server.

Packaging refuses a catalog/index scope mismatch or a catalog document absent from the selected active snapshot. Always pass the printed `--data-dir`; the old default index does not acquire newly imported chunks. A code-only GitHub deployment cannot introduce new private corpus assets: use the full bundle command above. Private originals, extracted text and indexes stay out of Git.

## Revisions and options

An ID defaults to a slug of the filename. To replace a document or change its role, reuse its ID and explicitly pass `--replace`:

```bash
./ingest-document.sh --file /absolute/path/revised-notice.pdf \
  --id laboratory-notice --role prof --replace \
  --title "Revised laboratory notice" --topic "laboratory notice"
```

Use `--base-data-dir PATH` when the desired base differs from your local profile, and `--output-data-dir PATH` to choose an unused output path. Run `./ingest-document.sh --help` for all options. Broad notices can omit `--term`; dated notices can use `2026-AUG`, `2027-JAN`, `2027-SUM`, or a human-readable term range.

`--topic` accepts literal phrases, not executable regular expressions. Repeat it for useful alternatives. Topic matches trigger a login/role prompt; they are not the permission mechanism. Retrieval, graph paths, citations and downloads enforce scope even if the user's wording does not match a topic. Newly imported documents require the usual successful OTP login, and disabling email authentication leaves them inaccessible.
