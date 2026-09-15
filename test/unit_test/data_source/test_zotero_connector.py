"""Unit tests for ZoteroConnector."""

from unittest.mock import MagicMock, patch

import pytest

from common.data_source.exceptions import (
    ConnectorMissingCredentialError,
    ConnectorValidationError,
    CredentialExpiredError,
    InsufficientPermissionsError,
)
from common.data_source.models import SlimDocument
from common.data_source.zotero_connector import ZoteroCheckpoint, ZoteroConnector, ZoteroCredentialsNotSetUpError

_ZOTERO_CREDS = {"zotero_api_key": "abc123"}
_WEBDAV_CREDS = {"zotero_api_key": "abc123", "webdav_username": "user", "webdav_password": "pw"}


def _pdf_item(key: str, filename: str = "paper.pdf", link_mode: str = "imported_file", date_modified: str = "2026-01-15T10:00:00Z", **extra_data):
    data = {
        "contentType": "application/pdf",
        "linkMode": link_mode,
        "filename": filename,
        "md5": "d41d8cd98f00b204e9800998ecf8427e",
        "dateModified": date_modified,
        "parentItem": "PARENT1",
        **extra_data,
    }
    return {"key": key, "version": 1, "data": data}


def _items_response(items: list[dict], total_results: int | None = None):
    resp = MagicMock()
    resp.json.return_value = items
    resp.headers = {"Total-Results": str(total_results if total_results is not None else len(items))}
    return resp


# ---------------------------------------------------------------------------
# __init__ / build_connector validation
# ---------------------------------------------------------------------------


@pytest.mark.p2
def test_init_rejects_bad_library_type():
    with pytest.raises(ConnectorValidationError, match="library_type"):
        ZoteroConnector(library_type="team", library_id="1")


@pytest.mark.p2
def test_init_rejects_missing_library_id():
    with pytest.raises(ConnectorValidationError, match="library_id"):
        ZoteroConnector(library_type="user", library_id="")


@pytest.mark.p2
def test_init_requires_webdav_url_for_webdav_storage():
    with pytest.raises(ConnectorValidationError, match="webdav_url"):
        ZoteroConnector(library_type="user", library_id="1", attachment_storage="webdav")


@pytest.mark.p2
def test_library_path_for_user_and_group():
    assert ZoteroConnector(library_type="user", library_id="42")._library_path == "users/42"
    assert ZoteroConnector(library_type="group", library_id="42")._library_path == "groups/42"


# ---------------------------------------------------------------------------
# load_credentials
# ---------------------------------------------------------------------------


@pytest.mark.p2
def test_load_credentials_missing_api_key_raises():
    connector = ZoteroConnector(library_type="user", library_id="1")
    with pytest.raises(ConnectorMissingCredentialError):
        connector.load_credentials({})


@pytest.mark.p1
def test_load_credentials_success():
    connector = ZoteroConnector(library_type="user", library_id="1")
    assert connector.load_credentials(_ZOTERO_CREDS) is None
    assert connector.api_key == "abc123"


@pytest.mark.p2
def test_load_credentials_webdav_requires_username_password():
    connector = ZoteroConnector(library_type="user", library_id="1", attachment_storage="webdav", webdav_url="https://dav.example.com")
    with pytest.raises(ConnectorMissingCredentialError):
        connector.load_credentials({"zotero_api_key": "abc123"})


@pytest.mark.p1
def test_load_credentials_webdav_builds_client():
    connector = ZoteroConnector(library_type="user", library_id="1", attachment_storage="webdav", webdav_url="https://dav.example.com")
    with patch("common.data_source.zotero_connector.WebDAVClient") as mock_client_cls:
        connector.load_credentials(_WEBDAV_CREDS)
    mock_client_cls.assert_called_once_with(base_url="https://dav.example.com", auth=("user", "pw"))


# ---------------------------------------------------------------------------
# validate_connector_settings
# ---------------------------------------------------------------------------


@pytest.mark.p2
def test_validate_without_credentials_raises():
    connector = ZoteroConnector(library_type="user", library_id="1")
    with pytest.raises(ZoteroCredentialsNotSetUpError):
        connector.validate_connector_settings()


@pytest.mark.p1
def test_validate_success():
    connector = ZoteroConnector(library_type="user", library_id="1")
    connector.api_key = "abc123"

    key_resp = MagicMock(status_code=200, ok=True)
    probe_resp = MagicMock(status_code=200, ok=True)

    with patch("common.data_source.zotero_connector.requests.get", side_effect=[key_resp, probe_resp]):
        connector.validate_connector_settings()  # should not raise


@pytest.mark.p2
def test_validate_invalid_key_raises_credential_expired():
    connector = ZoteroConnector(library_type="user", library_id="1")
    connector.api_key = "bad-key"

    key_resp = MagicMock(status_code=403, ok=False)

    with patch("common.data_source.zotero_connector.requests.get", return_value=key_resp), pytest.raises(CredentialExpiredError):
        connector.validate_connector_settings()


@pytest.mark.p2
def test_validate_library_forbidden_raises_insufficient_permissions():
    connector = ZoteroConnector(library_type="group", library_id="999")
    connector.api_key = "abc123"

    key_resp = MagicMock(status_code=200, ok=True)
    probe_resp = MagicMock(status_code=403, ok=False)

    with patch("common.data_source.zotero_connector.requests.get", side_effect=[key_resp, probe_resp]), pytest.raises(InsufficientPermissionsError):
        connector.validate_connector_settings()


@pytest.mark.p2
def test_validate_library_not_found_raises():
    connector = ZoteroConnector(library_type="group", library_id="999")
    connector.api_key = "abc123"

    key_resp = MagicMock(status_code=200, ok=True)
    probe_resp = MagicMock(status_code=404, ok=False)

    with patch("common.data_source.zotero_connector.requests.get", side_effect=[key_resp, probe_resp]), pytest.raises(ConnectorValidationError, match="not found"):
        connector.validate_connector_settings()


@pytest.mark.p2
def test_validate_webdav_probes_server():
    connector = ZoteroConnector(library_type="user", library_id="1", attachment_storage="webdav", webdav_url="https://dav.example.com")
    connector.api_key = "abc123"
    connector._webdav_client = MagicMock()

    key_resp = MagicMock(status_code=200, ok=True)
    probe_resp = MagicMock(status_code=200, ok=True)

    with patch("common.data_source.zotero_connector.requests.get", side_effect=[key_resp, probe_resp]):
        connector.validate_connector_settings()

    connector._webdav_client.exists.assert_called_once_with("/")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


@pytest.mark.p2
def test_build_dummy_checkpoint():
    connector = ZoteroConnector(library_type="user", library_id="1")
    ckpt = connector.build_dummy_checkpoint()
    assert isinstance(ckpt, ZoteroCheckpoint)
    assert ckpt.has_more is True
    assert ckpt.next_start == 0


@pytest.mark.p2
def test_validate_checkpoint_json_invalid_returns_dummy():
    connector = ZoteroConnector(library_type="user", library_id="1")
    ckpt = connector.validate_checkpoint_json("not-json")
    assert isinstance(ckpt, ZoteroCheckpoint)


# ---------------------------------------------------------------------------
# load_from_checkpoint: filtering + pagination
# ---------------------------------------------------------------------------


@pytest.mark.p1
def test_load_from_checkpoint_fetches_pdf_and_skips_non_pdf():
    connector = ZoteroConnector(library_type="user", library_id="1")
    connector.api_key = "abc123"

    items = [
        _pdf_item("KEY1"),
        {"key": "KEY2", "version": 1, "data": {"contentType": "text/html", "linkMode": "imported_url", "dateModified": "2026-01-15T10:00:00Z"}},
    ]

    with (
        patch("common.data_source.zotero_connector.request_with_retries", return_value=_items_response(items)),
        patch.object(connector, "_fetch_attachment_zotero_storage", return_value=b"%PDF-1.4 fake"),
    ):
        checkpoint = connector.build_dummy_checkpoint()
        docs = []
        gen = connector.load_from_checkpoint(0, 0, checkpoint)
        try:
            while True:
                docs.append(next(gen))
        except StopIteration as stop:
            final_checkpoint = stop.value

    assert len(docs) == 1
    assert docs[0].id == "zotero-attachment-KEY1"
    assert docs[0].extension == ".pdf"
    assert docs[0].metadata["parent_item"] == "PARENT1"
    assert final_checkpoint.has_more is False
    assert final_checkpoint.next_start == 0


@pytest.mark.p1
def test_load_from_checkpoint_skips_linked_file_and_linked_url():
    connector = ZoteroConnector(library_type="user", library_id="1")
    connector.api_key = "abc123"

    items = [
        _pdf_item("KEY1", link_mode="linked_file"),
        _pdf_item("KEY2", link_mode="linked_url"),
    ]

    with patch("common.data_source.zotero_connector.request_with_retries", return_value=_items_response(items)):
        checkpoint = connector.build_dummy_checkpoint()
        docs = list(_drain(connector.load_from_checkpoint(0, 0, checkpoint))[0])

    assert docs == []


@pytest.mark.p1
def test_load_from_checkpoint_paginates_via_next_start():
    connector = ZoteroConnector(library_type="user", library_id="1")
    connector.api_key = "abc123"

    page1 = _items_response([_pdf_item("KEY1")], total_results=2)
    checkpoint = connector.build_dummy_checkpoint()

    with patch("common.data_source.zotero_connector.request_with_retries", return_value=page1) as mock_req, patch.object(connector, "_fetch_attachment_zotero_storage", return_value=b"x"):
        _, next_checkpoint = _drain(connector.load_from_checkpoint(0, 0, checkpoint))

    assert next_checkpoint.has_more is True
    assert next_checkpoint.next_start == 1
    assert mock_req.call_args.kwargs["params"]["start"] == 0


@pytest.mark.p2
def test_load_from_checkpoint_filters_by_time_window():
    connector = ZoteroConnector(library_type="user", library_id="1")
    connector.api_key = "abc123"

    old_item = _pdf_item("OLD", date_modified="2020-01-01T00:00:00Z")
    new_item = _pdf_item("NEW", date_modified="2026-06-01T00:00:00Z")

    with (
        patch("common.data_source.zotero_connector.request_with_retries", return_value=_items_response([old_item, new_item])),
        patch.object(connector, "_fetch_attachment_zotero_storage", return_value=b"x"),
    ):
        checkpoint = connector.build_dummy_checkpoint()
        import datetime as dt

        start = dt.datetime(2025, 1, 1, tzinfo=dt.UTC).timestamp()
        docs, _ = _drain(connector.load_from_checkpoint(start, 0, checkpoint))

    assert [d.id for d in docs] == ["zotero-attachment-NEW"]


@pytest.mark.p1
def test_load_from_checkpoint_raises_on_listing_error():
    """A listing failure must propagate so the sync task fails loudly."""
    connector = ZoteroConnector(library_type="user", library_id="1")
    connector.api_key = "abc123"

    with patch("common.data_source.zotero_connector.request_with_retries", side_effect=RuntimeError("boom")):
        checkpoint = connector.build_dummy_checkpoint()
        with pytest.raises(RuntimeError, match="boom"):
            list(connector.load_from_checkpoint(0, 0, checkpoint))


@pytest.mark.p2
def test_load_from_checkpoint_yields_failure_on_attachment_fetch_error():
    connector = ZoteroConnector(library_type="user", library_id="1")
    connector.api_key = "abc123"

    with (
        patch("common.data_source.zotero_connector.request_with_retries", return_value=_items_response([_pdf_item("KEY1")])),
        patch.object(connector, "_fetch_attachment_zotero_storage", side_effect=RuntimeError("download failed")),
    ):
        checkpoint = connector.build_dummy_checkpoint()
        docs, failures, _ = _drain_all(connector.load_from_checkpoint(0, 0, checkpoint))

    assert docs == []
    assert len(failures) == 1
    assert failures[0].failed_document.document_id == "KEY1"


# ---------------------------------------------------------------------------
# _fetch_attachment_zotero_storage: size threshold + checksum
# ---------------------------------------------------------------------------


@pytest.mark.p2
def test_fetch_zotero_storage_rejects_oversized_content_length():
    connector = ZoteroConnector(library_type="user", library_id="1")
    connector.api_key = "abc123"

    resp = MagicMock()
    resp.headers = {"Content-Length": str(10**9)}

    with patch("common.data_source.zotero_connector.request_with_retries", return_value=resp), pytest.raises(ConnectorValidationError, match="size threshold"):
        connector._fetch_attachment_zotero_storage("KEY1", expected_md5=None)


@pytest.mark.p2
def test_fetch_zotero_storage_rejects_checksum_mismatch():
    connector = ZoteroConnector(library_type="user", library_id="1")
    connector.api_key = "abc123"

    resp = MagicMock()
    resp.headers = {"ETag": '"deadbeef"'}
    resp.iter_content.return_value = [b"hello"]

    with patch("common.data_source.zotero_connector.request_with_retries", return_value=resp), pytest.raises(ConnectorValidationError, match="checksum"):
        connector._fetch_attachment_zotero_storage("KEY1", expected_md5="notdeadbeef")


@pytest.mark.p1
def test_fetch_zotero_storage_returns_bytes_on_matching_checksum():
    connector = ZoteroConnector(library_type="user", library_id="1")
    connector.api_key = "abc123"

    resp = MagicMock()
    resp.headers = {"ETag": '"abc"'}
    resp.iter_content.return_value = [b"hel", b"lo"]

    with patch("common.data_source.zotero_connector.request_with_retries", return_value=resp):
        blob = connector._fetch_attachment_zotero_storage("KEY1", expected_md5="abc")

    assert blob == b"hello"


# ---------------------------------------------------------------------------
# _fetch_attachment_webdav: zip layout
# ---------------------------------------------------------------------------


@pytest.mark.p1
def test_fetch_attachment_webdav_extracts_single_member():
    import io
    import zipfile

    connector = ZoteroConnector(library_type="user", library_id="1", attachment_storage="webdav", webdav_url="https://dav.example.com")
    connector.api_key = "abc123"
    connector._webdav_client = MagicMock()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("paper.pdf", b"%PDF-1.4 content")
    zip_bytes = buf.getvalue()

    def _fake_download(remote_path, out_buffer):
        assert remote_path == "zotero/KEY1.zip"
        out_buffer.write(zip_bytes)

    connector._webdav_client.download_fileobj.side_effect = _fake_download

    blob = connector._fetch_attachment_webdav("KEY1")
    assert blob == b"%PDF-1.4 content"


@pytest.mark.p2
def test_fetch_attachment_webdav_rejects_multi_member_zip():
    import io
    import zipfile

    connector = ZoteroConnector(library_type="user", library_id="1", attachment_storage="webdav", webdav_url="https://dav.example.com")
    connector.api_key = "abc123"
    connector._webdav_client = MagicMock()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("a.pdf", b"a")
        zf.writestr("b.pdf", b"b")
    zip_bytes = buf.getvalue()

    connector._webdav_client.download_fileobj.side_effect = lambda _path, out: out.write(zip_bytes)

    with pytest.raises(ConnectorValidationError, match="exactly one file"):
        connector._fetch_attachment_webdav("KEY1")


@pytest.mark.p2
def test_fetch_attachment_webdav_requires_credentials():
    connector = ZoteroConnector(library_type="user", library_id="1", attachment_storage="webdav", webdav_url="https://dav.example.com")
    with pytest.raises(ZoteroCredentialsNotSetUpError):
        connector._fetch_attachment_webdav("KEY1")


# ---------------------------------------------------------------------------
# retrieve_all_slim_docs_perm_sync
# ---------------------------------------------------------------------------


@pytest.mark.p1
def test_retrieve_slim_docs_yields_slimdocument_batches():
    connector = ZoteroConnector(library_type="user", library_id="1")
    connector.api_key = "abc123"

    page1 = _items_response([_pdf_item("K1"), _pdf_item("K2")], total_results=3)
    page2 = _items_response([_pdf_item("K3")], total_results=3)

    with patch("common.data_source.zotero_connector.request_with_retries", side_effect=[page1, page2]):
        batches = list(connector.retrieve_all_slim_docs_perm_sync())

    flat = [item for batch in batches for item in batch]
    assert all(isinstance(item, SlimDocument) for item in flat)
    assert {item.id for item in flat} == {"zotero-attachment-K1", "zotero-attachment-K2", "zotero-attachment-K3"}


@pytest.mark.p2
def test_retrieve_slim_docs_skips_non_pdf():
    connector = ZoteroConnector(library_type="user", library_id="1")
    connector.api_key = "abc123"

    items = [_pdf_item("K1"), {"key": "K2", "data": {"contentType": "text/html", "linkMode": "imported_url"}}]
    with patch("common.data_source.zotero_connector.request_with_retries", return_value=_items_response(items)):
        batches = list(connector.retrieve_all_slim_docs_perm_sync())

    flat = [item for batch in batches for item in batch]
    assert [item.id for item in flat] == ["zotero-attachment-K1"]


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _drain(gen):
    """Drain a CheckpointOutput generator; return (documents, final_checkpoint)."""
    docs = []
    try:
        while True:
            docs.append(next(gen))
    except StopIteration as stop:
        return docs, stop.value


def _drain_all(gen):
    """Drain a CheckpointOutput generator; split Document/ConnectorFailure yields."""
    from common.data_source.models import ConnectorFailure, Document

    docs, failures = [], []
    try:
        while True:
            item = next(gen)
            if isinstance(item, Document):
                docs.append(item)
            elif isinstance(item, ConnectorFailure):
                failures.append(item)
    except StopIteration as stop:
        return docs, failures, stop.value
