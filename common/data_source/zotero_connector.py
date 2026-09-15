"""Zotero data source connector"""

import logging
import zipfile
from datetime import UTC, datetime
from io import BytesIO
from typing import Any

import requests
from webdav4.client import Client as WebDAVClient

from common.data_source.config import DOWNLOAD_CHUNK_SIZE, INDEX_BATCH_SIZE, REQUEST_TIMEOUT_SECONDS, ZOTERO_API_BASE_URL, ZOTERO_CONNECTOR_ATTACHMENT_SIZE_THRESHOLD, DocumentSource
from common.data_source.cross_connector_utils.retry_wrapper import request_with_retries
from common.data_source.exceptions import (
    ConnectorMissingCredentialError,
    ConnectorValidationError,
    CredentialExpiredError,
    InsufficientPermissionsError,
)
from common.data_source.interfaces import CheckpointedConnector, CheckpointOutput, SlimConnectorWithPermSync
from common.data_source.models import ConnectorCheckpoint, ConnectorFailure, Document, DocumentFailure, GenerateSlimDocumentOutput, SecondsSinceUnixEpoch, SlimDocument

logger = logging.getLogger(__name__)

_PAGE_SIZE = 100
_PDF_CONTENT_TYPE = "application/pdf"
# linked_file/linked_url attachments have no retrievable bytes; skip them.
_FETCHABLE_LINK_MODES = {"imported_file", "imported_url", "embedded_image"}
_SLIM_BATCH_SIZE = 1000


class ZoteroCredentialsNotSetUpError(PermissionError):
    def __init__(self) -> None:
        super().__init__("Zotero credentials are not set up, was load_credentials called?")


def _parse_zotero_datetime(value: str | None) -> datetime | None:
    """Parse a Zotero ``dateModified``/``dateAdded`` string (e.g. 2024-01-15T10:23:45Z)."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


class ZoteroCheckpoint(ConnectorCheckpoint):
    """Zotero-specific checkpoint: pagination offset within one sync call."""

    next_start: int = 0


class ZoteroConnector(SlimConnectorWithPermSync, CheckpointedConnector[ZoteroCheckpoint]):
    """Zotero connector: syncs PDF attachments from a user or group library."""

    def __init__(
        self,
        library_type: str,
        library_id: str,
        attachment_storage: str = "zotero",
        webdav_url: str | None = None,
        webdav_prefix: str = "zotero",
        batch_size: int = INDEX_BATCH_SIZE,
    ) -> None:
        if library_type not in ("user", "group"):
            raise ConnectorValidationError("Zotero library_type must be 'user' or 'group'.")
        if not library_id:
            raise ConnectorValidationError("Zotero library_id is required.")
        if attachment_storage not in ("zotero", "webdav"):
            raise ConnectorValidationError("Zotero attachment_storage must be 'zotero' or 'webdav'.")
        if attachment_storage == "webdav" and not webdav_url:
            raise ConnectorValidationError("Zotero webdav_url is required when attachment_storage is 'webdav'.")

        self.library_type = library_type
        self.library_id = str(library_id)
        self.attachment_storage = attachment_storage
        self.webdav_url = webdav_url.rstrip("/") if webdav_url else None
        self.webdav_prefix = (webdav_prefix or "").strip("/")
        self.batch_size = batch_size

        self.api_key: str | None = None
        self._webdav_client: WebDAVClient | None = None

    @property
    def _library_path(self) -> str:
        return f"{self.library_type}s/{self.library_id}"

    @classmethod
    def build_connector(cls, config: dict[str, Any]) -> "ZoteroConnector":
        connector = cls(
            library_type=config.get("library_type", "user"),
            library_id=config.get("library_id"),
            attachment_storage=config.get("attachment_storage", "zotero"),
            webdav_url=config.get("webdav_url"),
            webdav_prefix=config.get("webdav_prefix", "zotero"),
            batch_size=int(config.get("batch_size") or INDEX_BATCH_SIZE),
        )
        connector.load_credentials(config.get("credentials") or {})
        return connector

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def load_credentials(self, credentials: dict[str, Any]) -> dict[str, Any] | None:
        api_key = credentials.get("zotero_api_key")
        if not api_key:
            raise ConnectorMissingCredentialError("Zotero")
        self.api_key = api_key

        if self.attachment_storage == "webdav":
            username = credentials.get("webdav_username")
            password = credentials.get("webdav_password")
            if not username or not password:
                raise ConnectorMissingCredentialError("Zotero WebDAV attachment storage")
            self._webdav_client = WebDAVClient(base_url=self.webdav_url, auth=(username, password))
        return None

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ZoteroCredentialsNotSetUpError()
        return {"Zotero-API-Key": self.api_key, "Zotero-API-Version": "3"}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_connector_settings(self) -> None:
        if not self.api_key:
            raise ZoteroCredentialsNotSetUpError()

        try:
            key_resp = requests.get(f"{ZOTERO_API_BASE_URL}/keys/{self.api_key}", headers=self._headers(), timeout=REQUEST_TIMEOUT_SECONDS)
        except requests.RequestException as e:
            raise ConnectorValidationError(f"Could not reach the Zotero API: {e}") from e
        if key_resp.status_code in (401, 403):
            raise CredentialExpiredError("The Zotero API key is invalid or has been revoked.")
        if not key_resp.ok:
            raise ConnectorValidationError(f"Unexpected Zotero error validating the API key (status={key_resp.status_code}): {key_resp.text[:200]}")

        try:
            probe = requests.get(
                f"{ZOTERO_API_BASE_URL}/{self._library_path}/items",
                headers=self._headers(),
                params={"itemType": "attachment", "limit": 1},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException as e:
            raise ConnectorValidationError(f"Could not reach the Zotero {self.library_type} library: {e}") from e
        if probe.status_code in (401, 403):
            raise InsufficientPermissionsError(f"The Zotero API key cannot read {self._library_path}.")
        if probe.status_code == 404:
            raise ConnectorValidationError(f"Zotero {self.library_type} library '{self.library_id}' was not found.")
        if not probe.ok:
            raise ConnectorValidationError(f"Unexpected Zotero error probing the library (status={probe.status_code}): {probe.text[:200]}")

        if self.attachment_storage == "webdav":
            self._validate_webdav_settings()

    def _validate_webdav_settings(self) -> None:
        if self._webdav_client is None:
            raise ZoteroCredentialsNotSetUpError()
        try:
            self._webdav_client.exists("/")
        except Exception as e:
            raise ConnectorValidationError(f"Could not reach the Zotero WebDAV server: {e}") from e

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def build_dummy_checkpoint(self) -> ZoteroCheckpoint:
        return ZoteroCheckpoint(has_more=True, next_start=0)

    def validate_checkpoint_json(self, checkpoint_json: str) -> ZoteroCheckpoint:
        try:
            return ZoteroCheckpoint.model_validate_json(checkpoint_json)
        except Exception:
            return self.build_dummy_checkpoint()

    # ------------------------------------------------------------------
    # Item listing
    # ------------------------------------------------------------------

    def _fetch_items_page(self, start: int) -> tuple[list[dict[str, Any]], int]:
        resp = request_with_retries(
            "GET",
            f"{ZOTERO_API_BASE_URL}/{self._library_path}/items",
            headers=self._headers(),
            params={
                "itemType": "attachment",
                "start": start,
                "limit": _PAGE_SIZE,
                "sort": "dateModified",
                "direction": "asc",
            },
        )
        items = resp.json()
        total_results = int(resp.headers.get("Total-Results", start + len(items)))
        return items, total_results

    @staticmethod
    def _is_fetchable_pdf(data: dict[str, Any]) -> bool:
        return data.get("contentType") == _PDF_CONTENT_TYPE and data.get("linkMode") in _FETCHABLE_LINK_MODES

    # ------------------------------------------------------------------
    # Core data loading
    # ------------------------------------------------------------------

    def load_from_checkpoint(
        self,
        start: SecondsSinceUnixEpoch,
        end: SecondsSinceUnixEpoch,
        checkpoint: ZoteroCheckpoint,
    ) -> CheckpointOutput[ZoteroCheckpoint]:
        if not isinstance(checkpoint, ZoteroCheckpoint):
            checkpoint = self.build_dummy_checkpoint()
        checkpoint = checkpoint.model_copy(deep=True)

        # Listing failures propagate (fail the task); only per-attachment
        # fetch failures below are reported as soft ConnectorFailures.
        items, total_results = self._fetch_items_page(checkpoint.next_start)

        for item in items:
            data = item.get("data", {})
            if not self._is_fetchable_pdf(data):
                continue

            modified_at = _parse_zotero_datetime(data.get("dateModified"))
            if modified_at is not None:
                ts = modified_at.timestamp()
                if start and ts < start:
                    continue
                if end and ts > end:
                    continue

            key = item.get("key", "")
            try:
                document = self._attachment_to_document(item, data, modified_at)
            except Exception as e:
                logger.exception("Failed to fetch Zotero attachment %s", key)
                yield ConnectorFailure(
                    failed_document=DocumentFailure(document_id=key, document_link=data.get("url", "")),
                    failure_message=str(e),
                    exception=e,
                )
                continue
            yield document

        checkpoint.next_start += len(items)
        checkpoint.has_more = checkpoint.next_start < total_results
        if not checkpoint.has_more:
            checkpoint.next_start = 0
        return checkpoint

    def _attachment_to_document(self, item: dict[str, Any], data: dict[str, Any], modified_at: datetime | None) -> Document:
        key = item["key"]
        filename = data.get("filename") or f"{key}.pdf"

        if self.attachment_storage == "webdav":
            blob = self._fetch_attachment_webdav(key)
        else:
            blob = self._fetch_attachment_zotero_storage(key, expected_md5=data.get("md5"))

        metadata: dict[str, Any] = {"zotero_key": key, "link_mode": data.get("linkMode", "")}
        if parent_item := data.get("parentItem"):
            metadata["parent_item"] = parent_item
        if tags := data.get("tags"):
            tag_names = [tag.get("tag") for tag in tags if isinstance(tag, dict) and tag.get("tag")]
            if tag_names:
                metadata["tags"] = tag_names

        return Document(
            id=f"zotero-attachment-{key}",
            source=DocumentSource.ZOTERO,
            semantic_identifier=filename,
            extension=".pdf",
            blob=blob,
            size_bytes=len(blob),
            doc_updated_at=modified_at or datetime.now(UTC),
            fingerprint=data.get("md5"),
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Attachment byte fetch
    # ------------------------------------------------------------------

    def _fetch_attachment_zotero_storage(self, key: str, expected_md5: str | None) -> bytes:
        resp = request_with_retries(
            "GET",
            f"{ZOTERO_API_BASE_URL}/{self._library_path}/items/{key}/file",
            headers=self._headers(),
            stream=True,
        )
        content_length = resp.headers.get("Content-Length")
        if content_length and int(content_length) > ZOTERO_CONNECTOR_ATTACHMENT_SIZE_THRESHOLD:
            raise ConnectorValidationError(f"Attachment {key} ({content_length} bytes) exceeds the configured size threshold.")

        chunks: list[bytes] = []
        total = 0
        for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
            total += len(chunk)
            if total > ZOTERO_CONNECTOR_ATTACHMENT_SIZE_THRESHOLD:
                raise ConnectorValidationError(f"Attachment {key} exceeds the configured size threshold while streaming.")
            chunks.append(chunk)
        blob = b"".join(chunks)

        # Zotero's own docs recommend checking the ETag against the item's md5.
        etag = (resp.headers.get("ETag") or "").strip('"')
        if expected_md5 and etag and etag != expected_md5:
            raise ConnectorValidationError(f"Downloaded content for attachment {key} does not match its expected checksum.")
        return blob

    def _fetch_attachment_webdav(self, key: str) -> bytes:
        # Zotero's WebDAV layout (undocumented but stable): <key>.zip holds
        # exactly one member, the attachment file.
        if self._webdav_client is None:
            raise ZoteroCredentialsNotSetUpError()

        remote_path = f"{self.webdav_prefix}/{key}.zip" if self.webdav_prefix else f"{key}.zip"
        buffer = BytesIO()
        self._webdav_client.download_fileobj(remote_path, buffer)
        buffer.seek(0)

        with zipfile.ZipFile(buffer) as archive:
            members = [info for info in archive.infolist() if not info.is_dir()]
            if len(members) != 1:
                raise ConnectorValidationError(f"Unexpected Zotero WebDAV archive layout for {key}.zip: expected exactly one file, found {len(members)}.")
            member = members[0]
            if member.file_size > ZOTERO_CONNECTOR_ATTACHMENT_SIZE_THRESHOLD:
                raise ConnectorValidationError(f"Attachment {key} ({member.file_size} bytes) exceeds the configured size threshold.")
            with archive.open(member) as member_file:
                return member_file.read()

    # ------------------------------------------------------------------
    # Prune support
    # ------------------------------------------------------------------

    def retrieve_all_slim_docs_perm_sync(
        self,
        callback: Any = None,
    ) -> GenerateSlimDocumentOutput:
        del callback
        start = 0
        batch: list[SlimDocument] = []
        while True:
            items, total_results = self._fetch_items_page(start)
            if not items:
                break
            for item in items:
                data = item.get("data", {})
                if not self._is_fetchable_pdf(data):
                    continue
                batch.append(SlimDocument(id=f"zotero-attachment-{item['key']}"))
                if len(batch) >= _SLIM_BATCH_SIZE:
                    yield batch
                    batch = []
            start += len(items)
            if start >= total_results:
                break
        if batch:
            yield batch
