"""#49: each e-mail address in an answer is checked against the passages cited right after it."""
import pytest

from askiisc_serving.grounding import unsupported_emails

PROOF = {"prompt_marker": "S1", "chunk_ids": ["sched"], "text": "DS 252 taught by Ada Mora; office.cds@iisc.ac.in",
         "graph_support": [{"proof_id": "p", "contact_addresses": ["office.cds@iisc.ac.in", "ada.mora@example.edu"],
                            "required_chunk_ids": ["sched", "staff"]}]}
HOSTEL = {"prompt_marker": "S2", "chunk_ids": ["hostel"], "text": "Hostel queries: hostel.office@iisc.ac.in"}
STAFF = {"prompt_marker": "S3", "chunk_ids": ["staff"], "text": "Ada Mora ada.mora@example.edu; other.person@example.edu"}
UNRELATED = {"prompt_marker": "S4", "chunk_ids": ["unrel"], "text": "Arun Mora, wrong@example.edu"}


@pytest.mark.parametrize("answer, cited", [
    ("Contact office.cds@iisc.ac.in [S1] and hostel.office@iisc.ac.in [S2].", [PROOF, HOSTEL]),   # the #49 case
    ("The email is ada.mora@example.edu [S1][S3].", [PROOF, STAFF]),          # proved contact, text on the staff page
    ("Write to hostel.office@iisc.ac.in [S2].", [HOSTEL]),
    ("Hostel: hostel.office@iisc.ac.in. See [S2].", [HOSTEL]),                # no marker after it: all cited passages
])
def test_supported_addresses_are_accepted(answer, cited):
    assert unsupported_emails(answer, cited) == set()


@pytest.mark.parametrize("answer, cited, bad", [
    ("The email is guessed@example.edu [S1][S3].", [PROOF, STAFF], "guessed@example.edu"),
    ("Ada's email is other.person@example.edu [S1][S3].", [PROOF, STAFF], "other.person@example.edu"),
    ("Ada Mora's email is wrong@example.edu [S1][S4].", [PROOF, UNRELATED], "wrong@example.edu"),
    ("Write to wrong@example.edu [S2].", [HOSTEL, UNRELATED], "wrong@example.edu"),
])
def test_unsupported_or_misattributed_addresses_are_rejected(answer, cited, bad):
    assert unsupported_emails(answer, cited) == {bad}
