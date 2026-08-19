FROM python:3.12-slim
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg fonts-noto-core \
    && rm -rf /var/lib/apt/lists/*
COPY admin/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY admin /app/admin
COPY content /app/content
COPY deepstream /app/deepstream
COPY evaluation /app/evaluation
COPY validation/__init__.py /app/validation/__init__.py
COPY validation/strict_json.py /app/validation/strict_json.py
COPY validation/deepstream91_qualification.py /app/validation/deepstream91_qualification.py
COPY validation/product_acceptance_policy.py /app/validation/product_acceptance_policy.py
COPY validation/product_finalization_v2_contract.py /app/validation/product_finalization_v2_contract.py
COPY validation/open_video_manual_review.py /app/validation/open_video_manual_review.py
COPY validation/site_distance_evaluation.py /app/validation/site_distance_evaluation.py
COPY validation/site_distance_readiness_v2.py /app/validation/site_distance_readiness_v2.py
COPY validation/site_distance_evaluator_v2.py /app/validation/site_distance_evaluator_v2.py
COPY validation/schemas /app/validation/schemas
# Admin's exact-pin reader resolves static DS9 qualification anchors below
# DEEPSAFE_VALIDATION_WORKSPACE_ROOT.  Keep these three small immutable inputs
# in that namespace even when only generated validation results are mounted.
# Byte/size drift still fails closed against the pins compiled into
# admin/validation.py; no receipt or generated GPU evidence is baked in.
COPY deepstream/runtime-control-manifest.json /workspace/deepstream/runtime-control-manifest.json
COPY validation/ds9_runtime_compatibility.py /workspace/validation/ds9_runtime_compatibility.py
COPY validation/schemas/ds9-runtime-compatibility-v1.schema.json /workspace/validation/schemas/ds9-runtime-compatibility-v1.schema.json
# The read-only full-stack benchmark card independently replays this small,
# exact-pinned plan/validator/schema control bundle.  Generated receipts,
# videos and engines remain external and are never baked into the image.
COPY validation/full_stack_benchmark_contract_v1.py /workspace/validation/full_stack_benchmark_contract_v1.py
COPY validation/plans/deepstream-full-stack-benchmark-v1.json /workspace/validation/plans/deepstream-full-stack-benchmark-v1.json
COPY validation/schemas/deepstream-full-stack-benchmark-plan-v1.schema.json /workspace/validation/schemas/deepstream-full-stack-benchmark-plan-v1.schema.json
COPY validation/schemas/deepstream-full-stack-benchmark-receipt-v1.schema.json /workspace/validation/schemas/deepstream-full-stack-benchmark-receipt-v1.schema.json
ENV PYTHONPATH=/app
CMD ["uvicorn", "admin.app:app", "--host", "0.0.0.0", "--port", "8080"]
