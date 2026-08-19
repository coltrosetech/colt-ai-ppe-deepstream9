import asyncio
import hmac
import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from deepstream.config import (
    PERSON_PROFILES,
    list_person_profiles,
    module_readiness,
    resolve_person_profile,
    validate_module_selection,
)

from .pipeline import PipelineManager
from .store import JsonStore
from .endurance import load_endurance_status
from .manual_review import (
    ManualReviewError,
    ManualReviewService,
    parse_if_match,
)
from .ppe_human_qa import (
    PpeHumanQaError,
    PpeHumanQaService,
    parse_ppe_if_match,
)
from .roi_editor import RoiEditorService, create_roi_editor_router
from .roi_render_jobs import (
    RoiRenderJobService,
    create_roi_render_jobs_router,
)
from .validation import (
    ValidationArtifactError,
    load_validation_artifact,
    load_validation_status,
)

store = JsonStore()
pipeline = PipelineManager()
roi_editor = RoiEditorService()
roi_render_jobs = RoiRenderJobService(roi_editor)
changes = Counter("deepsafe_configuration_changes_total", "Configuration changes")


class SourceIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    uri: str = Field(pattern=r"^(rtsp|rtsps|file|http|https)://")


class Analytic(BaseModel):
    enabled: bool
    confidence: float = Field(ge=0, le=1)
    interval: int = Field(ge=0, le=60)


class AnalyticsIn(BaseModel):
    person: Analytic
    pose: Analytic
    ppe: Analytic


class InferenceIn(BaseModel):
    person_profile: str = Field(min_length=1, max_length=40)


class ManualDecisionIn(BaseModel):
    """Only human judgement is accepted; all linkage is server-owned."""

    status: Literal["reviewed", "ambiguous", "excluded"]
    detection_count_reviewed: int | None = Field(default=None, ge=0, le=1_000_000, strict=True)
    visible_person_count_confirmed: int | None = Field(default=None, ge=0, le=1_000_000, strict=True)
    scorable_person_count_confirmed: int | None = Field(default=None, ge=0, le=1_000_000, strict=True)
    true_positive_count: int | None = Field(default=None, ge=0, le=1_000_000, strict=True)
    false_positive_count: int | None = Field(default=None, ge=0, le=1_000_000, strict=True)
    false_negative_count: int | None = Field(default=None, ge=0, le=1_000_000, strict=True)
    ignored_detection_count: int | None = Field(default=None, ge=0, le=1_000_000, strict=True)
    unscorable_visible_person_count: int | None = Field(default=None, ge=0, le=1_000_000, strict=True)
    reasons: list[str] = Field(default_factory=list, max_length=16)
    reviewer_type: Literal["human", "human_with_ai_assist"] = "human"

    model_config = {"extra": "forbid"}


class PpeHumanQaSampleDecisionIn(BaseModel):
    decision: Literal[
        "approve_sample_semantics",
        "reject_sample_semantics",
        "uncertain_needs_relabel",
    ]
    reason_code: Literal[
        "visually_correct",
        "held_or_carried_helmet",
        "helmet_not_on_associated_head",
        "head_zone_policy_mismatch",
        "harness_not_hi_vis",
        "ordinary_garment_not_hi_vis",
        "no_helmet_box_wrong",
        "quarantine_should_be_retained",
        "retained_should_be_quarantined",
        "unknown_must_not_be_negative_ground_truth",
        "image_too_ambiguous",
        "other",
    ]
    notes: str = Field(default="", max_length=4000)

    model_config = {"extra": "forbid"}


class PpeHumanQaPolicyDecisionIn(BaseModel):
    decision: Literal["approve", "reject"]
    notes: str = Field(min_length=1, max_length=4000)

    model_config = {"extra": "forbid"}


def auth(authorization: str | None = Header(default=None)):
    if os.getenv("DEEPSAFE_ADMIN_AUTH_DISABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    token = os.getenv("DEEPSAFE_ADMIN_TOKEN", "change-me")
    if token != "change-me" and authorization != f"Bearer {token}":
        raise HTTPException(401, "Gecersiz yonetici tokeni")


def manual_review_auth(authorization: str | None = Header(default=None)):
    if os.getenv("DEEPSAFE_ADMIN_AUTH_DISABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return

    token = os.getenv("DEEPSAFE_ADMIN_TOKEN", "").strip()
    if not token or token == "change-me":
        raise HTTPException(503, "Manuel inceleme yetkilendirmesi yapilandirilmadi")
    supplied = "" if authorization is None else authorization
    if not hmac.compare_digest(supplied, f"Bearer {token}"):
        raise HTTPException(401, "Gecersiz yonetici tokeni")


def _manual_review_error(exc: ManualReviewError) -> HTTPException:
    labels = {
        "if_match_required": "If-Match revizyonu gerekli",
        "invalid_if_match": "Gecersiz If-Match revizyonu",
        "stale_revision": "Karar daha yeni bir revizyona sahip",
        "decision_not_found": "Inceleme karari bulunamadi",
        "asset_unavailable": "Inceleme varligi kullanilamiyor",
        "evidence_not_ready": "Bagli inceleme kanitlari henuz hazir degil",
        "invalid_decision": "Yalnizca nihai insan karari kaydedilebilir",
        "invalid_reviewer": "Gecersiz inceleyen kimligi",
        "decision_invariant_failed": "Karar sayim uzlasimlarini veya kaynak araliklarini saglamiyor",
    }
    return HTTPException(
        exc.status_code,
        labels.get(exc.state, "Manuel inceleme guvenli bicimde kullanilamiyor"),
    )


def _ppe_human_qa_error(exc: PpeHumanQaError) -> HTTPException:
    labels = {
        "if_match_required": "If-Match revizyonu gerekli",
        "invalid_if_match": "Gecersiz If-Match revizyonu",
        "stale_revision": "PPE karari daha yeni bir revizyona sahip",
        "sample_not_found": "PPE R6 ornegi bulunamadi",
        "policy_not_found": "PPE R6 politika karari bulunamadi",
        "tile_unavailable": "PPE R6 inceleme goruntusu kullanilamiyor",
        "invalid_decision": "PPE R6 insan karari gecersiz",
        "invalid_reviewer": "Gecersiz PPE inceleyen kimligi",
        "reviewer_identity_locked": "Bu PPE incelemesi baska bir inceleyen kimligine kilitli",
        "reviewer_not_initialized": "Export icin once inceleyen kimligiyle bir karar kaydedilmeli",
        "review_complete": "Tamamlanan PPE R6 incelemesi degistirilemez",
        "invalid_filter": "Gecersiz PPE R6 kuyruk filtresi",
        "packet_integrity_failed": "PPE R6 packet exact-pin butunlugu dogrulanamadi",
        "schema_integrity_failed": "PPE R6 strict sema butunlugu dogrulanamadi",
        "invalid_audit_store": "PPE R6 append-only denetim zinciri dogrulanamadi",
    }
    return HTTPException(
        exc.status_code,
        labels.get(exc.state, "PPE R6 insan QA guvenli bicimde kullanilamiyor"),
    )


@asynccontextmanager
async def lifespan(_app):
    roi_render_jobs.start()
    try:
        yield
    finally:
        roi_render_jobs.shutdown()
        pipeline.stop()


app = FastAPI(title="DeepSafe Vision", version="0.1.0", lifespan=lifespan)
app.include_router(create_roi_editor_router(roi_editor, auth))
app.include_router(create_roi_render_jobs_router(roi_render_jobs, auth))


@app.get("/", response_class=HTMLResponse)
def index():
    return (open(os.path.join(os.path.dirname(__file__), "static/index.html"), encoding="utf-8").read())


@app.get("/api/health")
def health(): return {"ok": True, "deepstream": pipeline.status(store.load())}


@app.get("/api/status")
def status():
    state = store.load()
    return pipeline.status(state) | {
        "state": state,
        "endurance_campaign": load_endurance_status(),
        "module_readiness": module_readiness(),
    }


@app.get("/api/endurance")
def endurance_status():
    return load_endurance_status()


@app.get("/api/validation")
def validation_status(artifact: str | None = None):
    """Read campaign summaries or one bounded, allow-listed evidence artifact."""

    if artifact is None:
        return load_validation_status()
    try:
        evidence = load_validation_artifact(artifact)
    except ValidationArtifactError as exc:
        status_code = 413 if exc.state == "too_large" else 404
        raise HTTPException(
            status_code, f"Dogrulama kaniti kullanilamiyor: {exc.state}"
        ) from exc
    return Response(
        evidence.content,
        media_type=evidence.media_type,
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": f'inline; filename="{evidence.filename}"',
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.get("/api/manual-review/queue", dependencies=[Depends(manual_review_auth)])
def manual_review_queue(response: Response):
    try:
        payload = ManualReviewService().queue()
    except ManualReviewError as exc:
        raise _manual_review_error(exc) from exc
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return payload


@app.get(
    "/api/manual-review/decisions/{decision_id}",
    dependencies=[Depends(manual_review_auth)],
)
def manual_review_detail(decision_id: str):
    try:
        payload, etag = ManualReviewService().detail(decision_id)
    except ManualReviewError as exc:
        raise _manual_review_error(exc) from exc
    return Response(
        content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        media_type="application/json",
        headers={"ETag": etag, "Cache-Control": "no-store"},
    )


@app.get(
    "/api/manual-review/assets/{asset_id}",
    dependencies=[Depends(manual_review_auth)],
)
def manual_review_asset(asset_id: str):
    try:
        asset = ManualReviewService().asset(asset_id)
    except ManualReviewError as exc:
        raise _manual_review_error(exc) from exc
    return Response(
        asset.content,
        media_type=asset.media_type,
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": f'inline; filename="{asset.filename}"',
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "default-src 'none'; sandbox",
        },
    )


@app.put(
    "/api/manual-review/decisions/{decision_id}",
    dependencies=[Depends(manual_review_auth)],
)
def update_manual_review_decision(
    decision_id: str,
    value: ManualDecisionIn,
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
    reviewer_id: Annotated[str | None, Header(alias="X-Reviewer-ID")] = None,
):
    try:
        expected_revision = parse_if_match(if_match)
        if reviewer_id is None:
            raise ManualReviewError("invalid_reviewer", status_code=422)
        decision = value.model_dump(exclude={"reviewer_type"})
        payload, etag = ManualReviewService().put(
            decision_id,
            expected_revision=expected_revision,
            reviewer_id=reviewer_id,
            reviewer_type=value.reviewer_type,
            decision=decision,
        )
    except ManualReviewError as exc:
        raise _manual_review_error(exc) from exc
    changes.inc()
    return Response(
        content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        media_type="application/json",
        headers={"ETag": etag, "Cache-Control": "no-store"},
    )


@app.get(
    "/api/ppe-human-qa-r6/queue",
    dependencies=[Depends(manual_review_auth)],
)
def ppe_human_qa_queue(
    response: Response,
    offset: int = 0,
    limit: int = 24,
    status: str | None = None,
    category: str | None = None,
    role: str | None = None,
    decision: str | None = None,
    query: str | None = None,
):
    try:
        payload = PpeHumanQaService().queue(
            offset=offset,
            limit=limit,
            status=status,
            category=category,
            role=role,
            decision=decision,
            query=query,
        )
    except PpeHumanQaError as exc:
        raise _ppe_human_qa_error(exc) from exc
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return payload


@app.get(
    "/api/ppe-human-qa-r6/samples/{sample_id}",
    dependencies=[Depends(manual_review_auth)],
)
def ppe_human_qa_sample(sample_id: str):
    try:
        payload, etag = PpeHumanQaService().sample(sample_id)
    except PpeHumanQaError as exc:
        raise _ppe_human_qa_error(exc) from exc
    return Response(
        content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        media_type="application/json",
        headers={
            "ETag": etag,
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.get(
    "/api/ppe-human-qa-r6/tiles/{sample_id}",
    dependencies=[Depends(manual_review_auth)],
)
def ppe_human_qa_tile(sample_id: str):
    try:
        tile = PpeHumanQaService().tile(sample_id)
    except PpeHumanQaError as exc:
        raise _ppe_human_qa_error(exc) from exc
    return Response(
        tile.content,
        media_type=tile.media_type,
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": f'inline; filename="{tile.filename}"',
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "default-src 'none'; sandbox",
        },
    )


@app.put(
    "/api/ppe-human-qa-r6/samples/{sample_id}",
    dependencies=[Depends(manual_review_auth)],
)
def update_ppe_human_qa_sample(
    sample_id: str,
    value: PpeHumanQaSampleDecisionIn,
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
    reviewer_id: Annotated[str | None, Header(alias="X-Reviewer-ID")] = None,
):
    try:
        expected_revision = parse_ppe_if_match(if_match)
        if reviewer_id is None:
            raise PpeHumanQaError("invalid_reviewer", status_code=422)
        payload, etag = PpeHumanQaService().put_sample(
            sample_id,
            expected_revision=expected_revision,
            reviewer_identity=reviewer_id,
            decision=value.decision,
            reason_code=value.reason_code,
            notes=value.notes,
        )
    except PpeHumanQaError as exc:
        raise _ppe_human_qa_error(exc) from exc
    changes.inc()
    return Response(
        content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        media_type="application/json",
        headers={
            "ETag": etag,
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.put(
    "/api/ppe-human-qa-r6/policies/{policy}",
    dependencies=[Depends(manual_review_auth)],
)
def update_ppe_human_qa_policy(
    policy: str,
    value: PpeHumanQaPolicyDecisionIn,
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
    reviewer_id: Annotated[str | None, Header(alias="X-Reviewer-ID")] = None,
):
    try:
        expected_revision = parse_ppe_if_match(if_match)
        if reviewer_id is None:
            raise PpeHumanQaError("invalid_reviewer", status_code=422)
        payload, etag = PpeHumanQaService().put_policy(
            policy,
            expected_revision=expected_revision,
            reviewer_identity=reviewer_id,
            decision=value.decision,
            notes=value.notes,
        )
    except PpeHumanQaError as exc:
        raise _ppe_human_qa_error(exc) from exc
    changes.inc()
    return Response(
        content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        media_type="application/json",
        headers={
            "ETag": etag,
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.get(
    "/api/ppe-human-qa-r6/export",
    dependencies=[Depends(manual_review_auth)],
)
def export_ppe_human_qa():
    try:
        payload = PpeHumanQaService().export()
    except PpeHumanQaError as exc:
        raise _ppe_human_qa_error(exc) from exc
    return Response(
        content=json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        media_type="application/json",
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": 'attachment; filename="ppe-human-qa-adjudication-r6.json"',
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.get("/api/sources")
def sources(): return store.load()["sources"]


@app.post("/api/sources", dependencies=[Depends(auth)])
def add_source(item: SourceIn):
    state = store.load()
    profile = resolve_person_profile(state)
    if len(state["sources"]) >= profile["max_batch_size"]:
        raise HTTPException(409, f"Secili motor en fazla {profile['max_batch_size']} kaynak destekliyor")
    source = item.model_dump() | {"id": str(uuid.uuid4())}
    state["sources"].append(source); store.save(state); changes.inc()
    return source


@app.delete("/api/sources/{source_id}", dependencies=[Depends(auth)])
def delete_source(source_id: str):
    state = store.load(); before = len(state["sources"])
    state["sources"] = [s for s in state["sources"] if s["id"] != source_id]
    if len(state["sources"]) == before: raise HTTPException(404, "Kaynak bulunamadi")
    store.save(state); changes.inc(); return {"deleted": source_id}


@app.get("/api/analytics")
def analytics(): return store.load()["analytics"]


@app.get("/api/inference")
def inference():
    state = store.load()
    return {"selected": resolve_person_profile(state), "available": list_person_profiles()}


@app.put("/api/inference", dependencies=[Depends(auth)])
def update_inference(value: InferenceIn):
    if value.person_profile not in PERSON_PROFILES:
        raise HTTPException(400, "Bilinmeyen insan algilama profili")
    state = store.load()
    profile = PERSON_PROFILES[value.person_profile]
    if len(state["sources"]) > profile["max_batch_size"]:
        raise HTTPException(409, "Kaynak sayisi secilen motorun batch kapasitesini asiyor")
    state["inference"]["person_profile"] = value.person_profile
    store.save(state); changes.inc()
    return {"selected": resolve_person_profile(state), "available": list_person_profiles()}


@app.put("/api/analytics", dependencies=[Depends(auth)])
def update_analytics(value: AnalyticsIn):
    state = store.load()
    candidate = value.model_dump()
    try:
        validate_module_selection(state | {"analytics": candidate})
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    state["analytics"] = candidate; store.save(state); changes.inc()
    return state["analytics"]


@app.post("/api/pipeline/start", dependencies=[Depends(auth)])
def start():
    try: return pipeline.start(store.load())
    except ValueError as exc: raise HTTPException(400, str(exc)) from exc


@app.post("/api/pipeline/stop", dependencies=[Depends(auth)])
def stop(): return pipeline.stop()


@app.get("/api/events")
async def events():
    async def stream():
        while True:
            yield f"data: {{\"type\":\"heartbeat\"}}\n\n"
            await asyncio.sleep(5)
    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/metrics")
def metrics(): return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
