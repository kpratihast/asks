# Adding IISc departments

The current pilot covers public academic content from CDS, CSA and ECE. Source selection is an ingestion setting. Changing it does not rewrite an existing index. Build a new snapshot, review its source coverage, and point a pilot app at that snapshot.

## Build the embedded-index pilot

From the checkout root, with the normal Ask IISc dependencies and model configuration:

```bash
export ASKIISC_SOURCE_DEPARTMENTS=CDS,CSA,ECE
export ASKIISC_DATA_DIR="$PWD/data/department-pilot"
./run-local.sh --prepare
./run-local.sh --port 8102
```

For an existing installation, copy the previous snapshot's **raw caches** (`raw/` and `admissions/` when present) into the new data directory first. Copy directories, not just their index JSON files. The build reuses those source files, fetches missing explicit pages, and regenerates a separate Qdrant index, graph and snapshot. It does not fetch optional crawl groups. To fetch those groups, run `python -m askiisc_retrieval.cli fetch` using the same environment before preparing the index. Where a source cannot be fetched, inspect the warning and check the actual per-department chunk counts before promoting the snapshot. The CSA site can reject automated fetches. In that case use the verified central admissions cache or an approved cached copy of its department page.

Existing installations using the private catalogue must have the matching originals and reviewed extracts under `.local/private/`. Use the existing secure document transfer or a private full release bundle. Department routing is separate from document authorization. The public pilot leaves the CDS role policy in place.

## Add a department

1. Add a stable department code, display name and unambiguous aliases to [the registry](config/departments.yaml). `ASKIISC_DEPARTMENT_REGISTRY` can point to another registry file. Restart the process after changing it. Add `course_prefixes` only when the department exclusively owns those prefixes. Shared prefixes stay unassigned.
2. Add reviewed pages to [the source list](sources.yaml) with the same `department` code and explicit `doc_type` and `scope`. For a bounded crawl, add a `crawls` entry with `department`, `seeds` on one approved host, `max_depth` and `max_pages`. Use `ASKIISC_SOURCES_FILE` to select an alternate source list.
3. Add the new code to `ASKIISC_SOURCE_DEPARTMENTS` and build into a fresh data directory. Shared IISc pages use `department: IISc`. The admissions normalizer tags recognized department panels and contact rows separately. Review unfamiliar page layouts and joint offerings before adding them as sources.
4. Check cited questions for the new department and existing departments. Include one comparison, an ambiguous programme question, and public and restricted document queries. Promote the candidate when each cited department and programme is correct.

## Cluster retrieval service

If the cluster uses `ASKIISC_BACKEND=ingestion`, configure the external ingestion pipeline's `seeds.json` and write the department code into its Qdrant payloads. In the **retrieval service** environment set `ASKIISC_DEPARTMENTS=CDS,CSA,ECE,IISc` to include those departments and shared records. That service does not read `sources.yaml` when opening its server-backed index. Use the same registry for serving and retrieval. This change has local adapter tests. The pilot does not require a live cluster Neo4j or PostgreSQL instance.

## Routing

General admissions questions use evidence tagged `IISc`. If the index has no relevant institute source, the app abstains. A named department gets its own candidates plus shared institute evidence. Requests naming multiple departments get candidates separately for each and require citations for each side. Graph traversal stays within the resolved department or institute scope. Other open campus queries and cross-department comparisons use vector retrieval. Ambiguous programme policy questions ask which department the user means. The department names and aliases apply to question routing and public versus restricted login hints. Per-chunk scopes enforce access.

The graph rules version changed. For AWS use a full bundle with a newly built index. The code-only deployment path retains its previous data and cannot publish the new graph or sources.
